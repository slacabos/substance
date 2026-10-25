import {
  Color,
  DataTexture,
  IcosahedronGeometry,
  LinearFilter,
  LinearMipmapLinearFilter,
  RGFormat,
  RepeatWrapping,
  UnsignedByteType,
  Vector3,
  Vector4,
  type BufferGeometry,
} from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import {
  MeshPhysicalNodeMaterial,
  type MeshPhysicalNodeMaterialParameters,
  type Node,
  type StorageBufferNode,
} from 'three/webgpu';
import {
  attribute,
  Fn,
  float,
  length,
  materialRoughness,
  max,
  min,
  mix,
  normalize,
  normalLocal,
  positionGeometry,
  pow,
  smoothstep,
  transformNormalToView,
  uniform,
  uniformArray,
  varyingProperty,
  vec3,
  vertexIndex,
} from 'three/tsl';
import { MAX_IMPULSES } from '../config';
import { bakeFoamLace } from '../core/foamLace';
import { OCEAN_WEATHER_PATCHES } from '../core/oceanWeather';
import { OCEAN_GUSTS } from '../core/oceanWind';
import { OCEAN_DEFAULTS } from '../core/oceanTuning';
import { SHALLOWS } from '../core/oceanShores';
import { DeformMode, oceanDeform, orbDeform, type DeformUniforms } from '../nodes/deform';
import { LACE_TILE_CELLS, oceanColor, oceanFoam, oceanSunLight } from '../nodes/ocean';
import { snoise } from '../nodes/noise';
import { driftBasis } from '../nodes/oceanSurface';
import { oceanWaveSurface, type OceanWaveFields } from '../nodes/oceanWaves';
import { baseN, baseR } from '../nodes/shape';

export { DeformMode, type DeformUniforms };

/** The colour of clear water a few millimetres deep over pale rock. */
const SHALLOW_WATER = vec3(0.16, 0.62, 0.66);
/** How much higher water standing on the coast's sand is drawn than the sim has it, orb radii. */
const WATER_LIFT = 0.0006;

/** How far the simulated waves' vertices lean toward their crests, per unit of slope (orb radii). */
const CREST_LEAN = 0.012;
/** The most a vertex leans, orb radii: well under half the desktop mesh's 0.0165 spacing. */
const CREST_SHIFT = 0.004;

/** What the ocean material needs beyond the shared uniforms: its FFT waves, the simulated surface, the spacing. */
export interface OceanSurfaceOptions {
  waves: OceanWaveFields;
  /** Per-vertex storage from OceanSim.renderBuffers. */
  sim: {
    surface: StorageBufferNode<'vec4'>;
    slope: StorageBufferNode<'vec4'>;
    drift: StorageBufferNode<'vec4'>;
    /** depth (negative on land), shore distance, the shelf's lag, foam grain */
    ground: StorageBufferNode<'vec4'>;
    driftBlend: Node<'float'>;
  };
  spacing: number;
  coastal: boolean;
}

const vec4s = (n: number) => Array.from({ length: n }, () => new Vector4());

let lace: DataTexture | undefined;

/** The baked foam lace (core/foamLace.ts), made once and shared by every ocean. */
function foamLaceTexture(): DataTexture {
  if (lace) return lace;
  const size = 512;
  lace = new DataTexture(bakeFoamLace(size, LACE_TILE_CELLS), size, size, RGFormat, UnsignedByteType);
  lace.wrapS = lace.wrapT = RepeatWrapping;
  lace.minFilter = LinearMipmapLinearFilter;
  lace.magFilter = LinearFilter;
  lace.generateMipmaps = true;
  lace.anisotropy = 4;
  lace.needsUpdate = true;
  return lace;
}

/** The deform material's uniforms; made first when something else, such as the ocean's sim, needs them too. */
export function createDeformUniforms(): DeformUniforms {
  const imp = vec4s(MAX_IMPULSES).map((v) => v.set(0, 1, 0, 0));
  const impB = vec4s(MAX_IMPULSES);
  const weather = vec4s(OCEAN_WEATHER_PATCHES).map((v) => v.set(0, 1, 0, 0));
  const weatherB = vec4s(OCEAN_WEATHER_PATCHES).map((v) => v.set(0.35, 0, 0, 0));
  const gusts = vec4s(OCEAN_GUSTS).map((v) => v.set(0, 1, 0, 0));
  return {
    uTime: uniform(0),
    uImp: uniformArray(imp, 'vec4'),
    uImpB: uniformArray(impB, 'vec4'),
    imp,
    impB,
    uPress: uniform(new Vector4(0, 0, 1, 0)),
    uWob: uniform(new Vector4(0, 1, 0, 0)),
    uWeather: uniformArray(weather, 'vec4'),
    uWeatherB: uniformArray(weatherB, 'vec4'),
    weather,
    weatherB,
    uGusts: uniformArray(gusts, 'vec4'),
    gusts,
    uDeep: uniform(new Color('#0b3553')),
    uShallow: uniform(new Color('#1a7695')),
    uFoamScroll: uniform(0),
    uOceanWavelength: uniform(OCEAN_DEFAULTS.peakWavelength),
    uOceanAmplitude: uniform(OCEAN_DEFAULTS.amplitude),
    uOceanChoppiness: uniform(OCEAN_DEFAULTS.choppiness),
    uOceanFoam: uniform(OCEAN_DEFAULTS.foam),
    uSunDir: uniform(new Vector3(0, 1, 0)),
    uSunColor: uniform(new Color(0, 0, 0)),
  };
}

/**
 * A MeshPhysicalNodeMaterial whose vertices are displaced on the GPU (see nodes/deform.ts). Water and slime
 * normals come from finite differences of the same displacement, so ripples and dents light correctly; the
 * ocean's come per pixel from its FFT waves.
 */
export function createDeformMaterial(
  params: MeshPhysicalNodeMaterialParameters,
  mode: DeformMode,
  ocean?: OceanSurfaceOptions,
  uniforms: DeformUniforms = createDeformUniforms(),
): { material: MeshPhysicalNodeMaterial; uniforms: DeformUniforms } {
  const material = new MeshPhysicalNodeMaterial(params);

  if (mode !== DeformMode.Ocean || !ocean) {
    material.positionNode = Fn(() => {
      const { pos, nrm } = orbDeform(normalize(positionGeometry), mode, uniforms);
      normalLocal.assign(nrm);
      return pos;
    })();
    return { material, uniforms };
  }

  const { waves, sim, spacing, coastal } = ocean;
  const wave = varyingProperty('float', 'vWave');
  const simWave = varyingProperty('float', 'vSimWave');
  const weatherAmount = varyingProperty('float', 'vWeather');
  const damping = varyingProperty('float', 'vDamping');
  const direction = varyingProperty('vec3', 'vDirection');
  const simGradient = varyingProperty('vec3', 'vSimGradient');
  const simulatedFoam = varyingProperty('float', 'vSimFoam');
  const turbulence = varyingProperty('float', 'vTurbulence');
  const objA = varyingProperty('vec3', 'vObjA');
  const objB = varyingProperty('vec3', 'vObjB');
  const shore = coastal ? varyingProperty('float', 'vShoreDistance') : undefined;
  // how deep the sim's own water stands here: over dry ground it's negative (its surface sinks under the ground)
  const column = coastal ? varyingProperty('float', 'vColumn') : undefined;
  material.positionNode = Fn(() => {
    const p = normalize(positionGeometry);
    const surface = oceanDeform(p, uniforms, waves, spacing, coastal);
    const simulated = sim.surface.element(vertexIndex);
    const simHeight = simulated.x;
    simGradient.assign(sim.slope.element(vertexIndex).xyz);
    normalLocal.assign(normalize(surface.nrm.sub(simGradient)));
    wave.assign(surface.height.add(simHeight));
    simWave.assign(simHeight);
    weatherAmount.assign(surface.weather);
    damping.assign(surface.damping);
    direction.assign(p);
    simulatedFoam.assign(simulated.y);
    turbulence.assign(simulated.z);
    // where the foam pattern is read: this vertex moved back by each of its two drifts
    const drift = sim.drift.element(vertexIndex);
    const { e1, e2 } = driftBasis(p);
    objA.assign(p.sub(e1.mul(drift.x)).sub(e2.mul(drift.y)));
    objB.assign(p.sub(e1.mul(drift.z)).sub(e2.mul(drift.w)));
    shore?.assign(attribute('oceanShoreData', 'vec4').w);

    // The simulated waves' crests sharpened as the swell's are: each vertex shifts a little uphill, so crests
    // narrow and troughs widen. Capped well under half the mesh spacing, so neighbours can never cross.
    const lean = simGradient.mul(CREST_LEAN);
    const shift = lean.mul(min(float(1), float(CREST_SHIFT).div(max(length(lean), 1e-6))));
    // (water standing on the coast's sand rides a hair higher than the sim's, for the same reason the seabed is
    // drawn a little deeper: the land's finer mesh would poke through a thin sheet)
    const depth = sim.ground.element(vertexIndex).x.add(simHeight);
    column?.assign(depth);
    const lift = coastal ? smoothstep(0, 0.002, depth).mul(WATER_LIFT) : float(0);
    return surface.pos.add(normalize(surface.pos).mul(simHeight.add(lift))).add(shift);
  })();

  // Per pixel: the full-resolution wave slopes, crests where the surface bunches up, and roughness for the
  // slopes too small to see.
  const p = normalize(direction);
  const surface = oceanWaveSurface(p, waves, uniforms, weatherAmount);
  // whitecaps from folding now and in the last few seconds
  const breaking = max(
    float(1).sub(smoothstep(0.3, 0.85, surface.jacobian)),
    smoothstep(0.6, 1.05, surface.foam),
  );
  const seaFoam = oceanFoam(
    { a: objA, b: objB, blend: sim.driftBlend },
    foamLaceTexture(),
    breaking,
    weatherAmount,
    simulatedFoam,
    uniforms,
    shore,
  );
  // where the sim's water thins out over sand its front is a band of bubbles, as a wave's run-up is: a broken,
  // ragged line rather than the mesh's own edge
  const edge = column
    ? float(1)
        .sub(smoothstep(0.0002, 0.0025, column))
        .mul(smoothstep(-0.35, 0.25, snoise(p.mul(380)).add(snoise(p.mul(1100)).mul(0.5))))
        .mul(0.9)
    : float(0);
  const foam = max(seaFoam, edge);
  const base = materialRoughness.add(turbulence.mul(0.1));
  // GGX alpha^2 grows by twice the hidden slope variance
  const rough = pow(pow(base, 4).add(surface.lostVariance.mul(damping).mul(damping).mul(2)), 0.25);
  const normal = transformNormalToView(
    normalize(
      baseN(p, p.mul(baseR(p)))
        .sub(surface.gradient.mul(damping).div(baseR(p)))
        .sub(simGradient),
    ),
  );
  const roughness = mix(rough, 0.92, foam);
  material.normalNode = normal;
  // Over a coast's shelf the water is a few millimetres deep: clear and turquoise, the sand showing through it,
  // turning to the open sea's deep blue only as the shelf runs out.
  // (and clearer still where the sim's own water thins out over sand: at a wave's edge on the beach, or over a
  // bar the sea has drawn back from)
  const thin = column ? float(1).sub(smoothstep(0.0004, 0.004, column)) : float(0);
  const shallow = shore ? max(float(1).sub(smoothstep(0.05, SHALLOWS, shore)), thin) : float(0);
  // where the water runs out it ends, by its own depth, rather than wherever its surface dips under the sand
  // (dry ground next to water carries the water's level just under the sand: the sand there hides it)
  if (column) material.maskNode = column.greaterThan(-0.004);
  const color = oceanColor(wave, simWave, foam, weatherAmount, uniforms);
  material.colorNode = mix(color, SHALLOW_WATER, shallow.mul(float(1).sub(foam)).mul(0.55));
  material.roughnessNode = roughness;
  material.emissiveNode = oceanSunLight(normal, wave, foam, roughness, uniforms);
  material.specularIntensityNode = mix(float(0.75), float(0.08), foam);
  material.transmissionNode = mix(mix(float(0.22), float(0.85), shallow), float(0), foam);
  // refraction through the shallows reaches no deeper than they are, a few millimetres
  material.thicknessNode = mix(float(0.7), float(0.004), shallow);
  return { material, uniforms };
}

/** A dense, indexed icosphere. Only positions matter: the shader rebuilds normals. */
export function createDeformGeometry(coarse: boolean, detail = coarse ? 44 : 72): BufferGeometry {
  const ico = new IcosahedronGeometry(1, detail);
  ico.deleteAttribute('uv');
  return mergeVertices(ico);
}
