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
  materialRoughness,
  max,
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
  vertexIndex,
} from 'three/tsl';
import { MAX_IMPULSES } from '../config';
import { bakeFoamLace } from '../core/foamLace';
import { OCEAN_WEATHER_PATCHES } from '../core/oceanWeather';
import { OCEAN_GUSTS } from '../core/oceanWind';
import { OCEAN_DEFAULTS } from '../core/oceanTuning';
import { DeformMode, oceanDeform, orbDeform, type DeformUniforms } from '../nodes/deform';
import { LACE_TILE_CELLS, oceanColor, oceanFoam, oceanSunLight } from '../nodes/ocean';
import { driftBasis } from '../nodes/oceanSurface';
import { oceanWaveSurface, type OceanWaveFields } from '../nodes/oceanWaves';
import { baseN, baseR } from '../nodes/shape';

export { DeformMode, type DeformUniforms };

/** What the ocean material needs beyond the shared uniforms: its FFT waves, the simulated surface, the spacing. */
export interface OceanSurfaceOptions {
  waves: OceanWaveFields;
  /** Per-vertex storage from OceanSim.renderBuffers. */
  sim: {
    surface: StorageBufferNode<'vec4'>;
    slope: StorageBufferNode<'vec4'>;
    drift: StorageBufferNode<'vec4'>;
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
    uOceanWind: uniform(OCEAN_DEFAULTS.windSpeed),
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
  const weatherAmount = varyingProperty('float', 'vWeather');
  const damping = varyingProperty('float', 'vDamping');
  const direction = varyingProperty('vec3', 'vDirection');
  const simGradient = varyingProperty('vec3', 'vSimGradient');
  const simulatedFoam = varyingProperty('float', 'vSimFoam');
  const turbulence = varyingProperty('float', 'vTurbulence');
  const objA = varyingProperty('vec3', 'vObjA');
  const objB = varyingProperty('vec3', 'vObjB');
  const shore = coastal ? varyingProperty('float', 'vShoreDistance') : undefined;
  material.positionNode = Fn(() => {
    const p = normalize(positionGeometry);
    const surface = oceanDeform(p, uniforms, waves, spacing, coastal);
    const simulated = sim.surface.element(vertexIndex);
    const simHeight = simulated.x;
    simGradient.assign(sim.slope.element(vertexIndex).xyz);
    normalLocal.assign(normalize(surface.nrm.sub(simGradient)));
    wave.assign(surface.height.add(simHeight));
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
    return surface.pos.add(normalize(surface.pos).mul(simHeight));
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
  const foam = oceanFoam(
    { a: objA, b: objB, blend: sim.driftBlend },
    foamLaceTexture(),
    breaking,
    weatherAmount,
    simulatedFoam,
    uniforms,
    shore,
  );
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
  material.colorNode = oceanColor(wave, foam, weatherAmount, uniforms);
  material.roughnessNode = roughness;
  material.emissiveNode = oceanSunLight(normal, wave, foam, roughness, uniforms);
  material.specularIntensityNode = mix(float(0.75), float(0.08), foam);
  material.transmissionNode = mix(float(0.22), float(0), foam);
  return { material, uniforms };
}

/** A dense, indexed icosphere. Only positions matter: the shader rebuilds normals. */
export function createDeformGeometry(coarse: boolean, detail = coarse ? 44 : 72): BufferGeometry {
  const ico = new IcosahedronGeometry(1, detail);
  ico.deleteAttribute('uv');
  return mergeVertices(ico);
}
