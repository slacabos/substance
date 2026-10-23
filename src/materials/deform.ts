import { Color, IcosahedronGeometry, Vector4, type BufferGeometry } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { MeshPhysicalNodeMaterial, type MeshPhysicalNodeMaterialParameters } from 'three/webgpu';
import {
  attribute,
  Fn,
  float,
  materialRoughness,
  mix,
  normalize,
  normalLocal,
  positionGeometry,
  uniform,
  uniformArray,
  varyingProperty,
} from 'three/tsl';
import { MAX_IMPULSES } from '../config';
import { OCEAN_WEATHER_PATCHES } from '../core/oceanWeather';
import { OCEAN_DEFAULTS } from '../core/oceanTuning';
import { DeformMode, orbDeform, type DeformUniforms } from '../nodes/deform';
import { oceanColor, oceanFineNormal, oceanFoam } from '../nodes/ocean';

export { DeformMode, type DeformUniforms };

const vec4s = (n: number) => Array.from({ length: n }, () => new Vector4());

/**
 * A MeshPhysicalNodeMaterial whose vertices are displaced on the GPU (see nodes/deform.ts).
 * Normals come from finite differences of the same displacement, so ripples and dents light correctly.
 */
export function createDeformMaterial(
  params: MeshPhysicalNodeMaterialParameters,
  mode: DeformMode,
): { material: MeshPhysicalNodeMaterial; uniforms: DeformUniforms } {
  const material = new MeshPhysicalNodeMaterial(params);
  const imp = vec4s(MAX_IMPULSES).map((v) => v.set(0, 1, 0, 0));
  const impB = vec4s(MAX_IMPULSES);
  const weather = vec4s(OCEAN_WEATHER_PATCHES).map((v) => v.set(0, 1, 0, 0));
  const weatherB = vec4s(OCEAN_WEATHER_PATCHES).map((v) => v.set(0.35, 0, 0, 0));
  const uniforms: DeformUniforms = {
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
    uDeep: uniform(new Color('#0b3553')),
    uShallow: uniform(new Color('#1a7695')),
    uOceanWind: uniform(OCEAN_DEFAULTS.windSpeed),
    uOceanWavelength: uniform(OCEAN_DEFAULTS.peakWavelength),
    uOceanAmplitude: uniform(OCEAN_DEFAULTS.amplitude),
    uOceanChoppiness: uniform(OCEAN_DEFAULTS.choppiness),
    uOceanFoam: uniform(OCEAN_DEFAULTS.foam),
  };

  const wave = varyingProperty('float', 'vWave');
  const breaking = varyingProperty('float', 'vBreaking');
  const weatherAmount = varyingProperty('float', 'vWeather');
  const simulatedFoam = varyingProperty('float', 'vSimFoam');
  const turbulence = varyingProperty('float', 'vTurbulence');
  const obj = varyingProperty('vec3', 'vObj');
  material.positionNode = Fn(() => {
    const p = normalize(positionGeometry);
    const { pos, nrm, height, breaking: breakAmount, weather: localWeather } = orbDeform(p, mode, uniforms);
    if (mode === DeformMode.Ocean) {
      const simHeight = attribute('oceanHeight', 'float');
      normalLocal.assign(normalize(nrm.sub(attribute('oceanGradient', 'vec3'))));
      wave.assign(height.add(simHeight));
      breaking.assign(breakAmount);
      weatherAmount.assign(localWeather);
      simulatedFoam.assign(attribute('oceanFoam', 'float'));
      turbulence.assign(attribute('oceanTurbulence', 'float'));
      obj.assign(p.sub(attribute('oceanDrift', 'vec3')));
      return pos.add(normalize(pos).mul(simHeight));
    }
    normalLocal.assign(nrm);
    return pos;
  })();

  if (mode === DeformMode.Ocean) {
    const foam = oceanFoam(obj, breaking, weatherAmount, simulatedFoam, uniforms);
    material.colorNode = oceanColor(wave, foam, weatherAmount, uniforms);
    material.roughnessNode = mix(materialRoughness.add(turbulence.mul(0.1)), 0.92, foam);
    material.specularIntensityNode = mix(float(0.75), float(0.08), foam);
    material.transmissionNode = mix(float(0.22), float(0), foam);
    material.normalNode = oceanFineNormal(obj, turbulence, uniforms);
  }
  return { material, uniforms };
}

/** A dense, indexed icosphere. Only positions matter: the shader rebuilds normals. */
export function createDeformGeometry(coarse: boolean, detail = coarse ? 44 : 72): BufferGeometry {
  const ico = new IcosahedronGeometry(1, detail);
  ico.deleteAttribute('uv');
  return mergeVertices(ico);
}
