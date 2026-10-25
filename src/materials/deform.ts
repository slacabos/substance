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
  vec2,
  vec3,
  vec4,
  vertexIndex,
  cameraFar,
  cameraViewMatrix,
  clamp,
  dot,
  cameraNear,
  linearDepth,
  viewportLinearDepth,
} from 'three/tsl';
import { MAX_IMPULSES } from '../config';
import { bakeFoamLace } from '../core/foamLace';
import { OCEAN_WEATHER_PATCHES } from '../core/oceanWeather';
import { OCEAN_GUSTS } from '../core/oceanWind';
import { OCEAN_DEFAULTS } from '../core/oceanTuning';
import { OCEAN_DEPTH } from '../core/oceanFFT';
import { DeformMode, oceanDeform, orbDeform, type DeformUniforms } from '../nodes/deform';
import { FOAM_LAYERS, LACE_TILE_CELLS, oceanColor, oceanFoam, oceanSunLight } from '../nodes/ocean';
import { SAND_KEY_LIGHT } from '../nodes/sand';
import { driftBasis } from '../nodes/oceanSurface';
import { oceanWaveSurface, type OceanWaveFields } from '../nodes/oceanWaves';
import { baseN, baseR } from '../nodes/shape';

export { DeformMode, type DeformUniforms };

/**
 * What's left of light after ABSORB_DISTANCE of coastal water (linear): red goes first, but not so much that
 * orange sand under a centimetre of it turns green; it browns, then blues as the water deepens.
 */
const SHALLOW_ABSORB = vec3(0.58, 0.76, 0.86);
/** The colour shallow coastal water scatters back (linear), before the deep sea's own colour takes over. */
const SHALLOW_SCATTER = vec3(0.015, 0.13, 0.24);
const ABSORB_DISTANCE = 0.01;
/** How bright the coast's foam's key light is, as the land's (KEY_STRENGTH in materials/oceanShores.ts). */
const FOAM_KEY = 1.4;
/** How much higher water standing on the coast's sand is drawn than the sim has it, orb radii. */
const WATER_LIFT = 0.0006;
/** Over how much water between it and the sand behind (view distance, orb radii) the water's edge fades in. */
const EDGE_FADE = [0.0002, 0.005] as const;

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
    /** the current (xyz), which drags the surf's foam into streaks */
    flow: StorageBufferNode<'vec4'>;
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
    uFoamWhitecap: uniform(FOAM_LAYERS.whitecap.coverage),
    uFoamAmbient: uniform(FOAM_LAYERS.ambient.coverage),
    uFoamShore: uniform(FOAM_LAYERS.shore.coverage),
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
  // the coast's values, packed in one varying (WebGPU passes at most 16 from vertex to fragment): distance to
  // shore, the water's column, the depth its colour goes by, and the surf's breaking
  const coast = varyingProperty('vec4', 'vCoast');
  // and which way the water runs, in the vertex's drift basis (driftBasis)
  const current = varyingProperty('vec2', 'vCurrent');
  const shore = coastal ? coast.x : undefined;
  // how deep the sim's own water stands here: over dry ground it's negative (its surface sinks under the ground)
  const column = coastal ? coast.y : undefined;
  const tintDepth = coastal ? coast.z : undefined;
  // and how hard the surf is breaking here this moment: a wave's fresh white front
  const impact = coastal ? coast.w : float(0);
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
    if (coastal) {
      // the way the surf's foam is dragged: up and down the beach, as the waves and the backwash run (the shore's
      // normal), or with the current where that runs fast (a rip; the wind's slow drift doesn't count)
      const flow = sim.flow.element(vertexIndex).xyz;
      const seaward = attribute('oceanShoreData', 'vec4').xyz;
      const speed = length(flow);
      const drag = mix(seaward, flow.div(max(speed, 1e-6)), smoothstep(0.2, 0.5, speed));
      current.assign(vec2(dot(drag, e1), dot(drag, e2)));
    }
    objA.assign(p.sub(e1.mul(drift.x)).sub(e2.mul(drift.y)));
    objB.assign(p.sub(e1.mul(drift.z)).sub(e2.mul(drift.w)));

    // The simulated waves' crests sharpened as the swell's are: each vertex shifts a little uphill, so crests
    // narrow and troughs widen. Capped well under half the mesh spacing, so neighbours can never cross.
    const lean = simGradient.mul(CREST_LEAN);
    const shift = lean.mul(min(float(1), float(CREST_SHIFT).div(max(length(lean), 1e-6))));
    // (water on the coast rides a hair higher than the sim's, for the same reason the seabed is drawn a little
    // deeper: the land's finer mesh would poke through a thin sheet. It's the same everywhere, so the water stays
    // one smooth surface up to its edge; a thin film fades out by the depth to the sand behind it, below)
    const depth = sim.ground.element(vertexIndex).x.add(simHeight);
    // the depth its colour goes by: the still water's, with only some of the waves riding on it, or every surf crest
    // would ring the shallows in bands of colour
    if (coastal)
      coast.assign(
        vec4(
          attribute('oceanShoreData', 'vec4').w,
          depth,
          sim.ground.element(vertexIndex).x.add(simHeight.mul(0.3)),
          simulated.w,
        ),
      );
    const lift = coastal ? float(WATER_LIFT) : float(0);
    // nor do crests lean near the water's edge, where a sideways shift would show as teeth along the beach
    const leaning = coastal ? smoothstep(0.004, 0.012, depth) : float(1);
    return surface.pos.add(normalize(surface.pos).mul(simHeight.add(lift))).add(shift.mul(leaning));
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
  // the way the water runs here (unit), or along the shore where it's still
  const basis = driftBasis(p);
  const alongCurrent = normalize(
    basis.e1.mul(current.x).add(basis.e2.mul(current.y)).add(basis.e1.mul(1e-4)),
  );
  const seaFoam = oceanFoam(
    { a: objA, b: objB, blend: sim.driftBlend },
    foamLaceTexture(),
    breaking,
    weatherAmount,
    simulatedFoam,
    uniforms,
    shore,
    coastal ? alongCurrent : undefined,
    impact,
  );
  // Where the water meets the sand: how far the sand behind lies from it (from the depth buffer, so along the
  // sand's own contours rather than the water's triangles). Over its last couple of millimetres the water fades
  // to a clear film.
  const behind = coastal ? viewportLinearDepth.sub(linearDepth()).mul(cameraFar.sub(cameraNear)) : float(1);
  const meets = smoothstep(EDGE_FADE[0], EDGE_FADE[1], behind);
  // (no foam line of its own there: the run-up drawn on the sand carries the foam on up the beach, and a line
  // here showed as a second white strike, cut off from it)
  // and the sea's own foam thins out over the last few millimetres too, handing over to the run-up's: foam
  // piled at the very edge drew it as a white rim
  const foam = seaFoam.amount.mul(coastal ? smoothstep(EDGE_FADE[0], EDGE_FADE[1] * 0.7, behind) : float(1));
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
  // Over the coast the water is coloured by how deep it is, as real water is: light through it is absorbed red
  // first (Beer-Lambert, the material's attenuation over the column as its thickness), so a film over the sand
  // barely tints it, a centimetre turns it turquoise and the open sea is deep blue; only deep water shows its own
  // body colour. Its edge needs no line: the colour just thins out with the water.
  const depth = tintDepth ? clamp(tintDepth, 0, OCEAN_DEPTH) : float(0);
  // how much the water's own colour shows over what's under it: light it scatters back, turquoise a centimetre
  // or two deep, the open sea's blue deeper still
  const body = column ? smoothstep(0.003, 0.04, depth) : float(1);
  // where the water runs out it ends, by its own depth, rather than wherever its surface dips under the sand
  // (dry ground next to water carries the water's level just under the sand, however steeply the sand rises
  // there, and the sand hides it; ground further from water sinks its surface DRY_SINK under, which this cuts)
  if (column) material.maskNode = column.greaterThan(-0.015);
  const color = oceanColor(wave, simWave, foam, weatherAmount, uniforms, seaFoam.color);
  // (transmitted light is tinted by the colour, so over the shallows the colour is clear and the absorption alone
  // tints it)
  const water = column
    ? mix(mix(SHALLOW_SCATTER, color, smoothstep(0.02, OCEAN_DEPTH, depth)), color, foam)
    : color;
  material.colorNode = mix(vec3(1), water, max(body.mul(meets), foam));
  material.roughnessNode = roughness;
  // (the coast's foam takes the same steady key light as its sand, nodes/shore.ts, so white water reads white in
  // every setting rather than turning the night sky's blue-grey beside bright sand)
  const sun = uniforms.uSunColor as unknown as Node<'vec3'>;
  const keyColor = sun.div(max(max(sun.x, sun.y), max(sun.z, 1e-3))).mul(FOAM_KEY);
  const keyDir = normalize(
    cameraViewMatrix.mul(vec4(SAND_KEY_LIGHT.x, SAND_KEY_LIGHT.y, SAND_KEY_LIGHT.z, 0)).xyz,
  );
  const foamKey = coastal
    ? keyColor
        .mul(max(dot(normal, keyDir), 0).mul(0.6).add(0.4))
        .mul(foam)
        .mul(1 / Math.PI)
    : vec3(0);
  material.emissiveNode = oceanSunLight(normal, wave, foam, roughness, uniforms)
    .mul(max(meets, foam))
    .add(foamKey);
  material.specularIntensityNode = mix(float(0.75), float(0.08), foam).mul(max(meets, foam));
  // (foam stays opaque white even at the very edge)
  material.transmissionNode = mix(mix(float(1), mix(float(0.97), float(0.22), body), meets), float(0), foam);
  if (coastal) {
    material.thicknessNode = depth;
    material.attenuationColorNode = SHALLOW_ABSORB;
    material.attenuationDistanceNode = float(ABSORB_DISTANCE);
  }
  return { material, uniforms };
}

/** A dense, indexed icosphere. Only positions matter: the shader rebuilds normals. */
export function createDeformGeometry(coarse: boolean, detail = coarse ? 44 : 72): BufferGeometry {
  const ico = new IcosahedronGeometry(1, detail);
  ico.deleteAttribute('uv');
  return mergeVertices(ico);
}
