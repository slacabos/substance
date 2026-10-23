import { Color, IcosahedronGeometry, Vector4, type BufferGeometry } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { MeshPhysicalNodeMaterial, type MeshPhysicalNodeMaterialParameters } from 'three/webgpu';
import {
  float,
  Fn,
  materialRoughness,
  mix,
  normalize,
  normalLocal,
  positionGeometry,
  smoothstep,
  uniform,
  uniformArray,
  varyingProperty,
} from 'three/tsl';
import { MAX_IMPULSES } from '../config';
import { DeformMode, orbDeform, type DeformUniforms } from '../nodes/deform';
import { oceanColor, oceanFoam } from '../nodes/ocean';

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
  const uniforms: DeformUniforms = {
    uTime: uniform(0),
    uImp: uniformArray(imp, 'vec4'),
    uImpB: uniformArray(impB, 'vec4'),
    imp,
    impB,
    uPress: uniform(new Vector4(0, 0, 1, 0)),
    uWob: uniform(new Vector4(0, 1, 0, 0)),
    uStorm: uniform(1),
    uDeep: uniform(new Color('#021a2e')),
    uShallow: uniform(new Color('#0b6f85')),
    uScatter: uniform(new Color('#1fb5a8')),
  };

  const wave = varyingProperty('float', 'vWave');
  const obj = varyingProperty('vec3', 'vObj');
  material.positionNode = Fn(() => {
    const p = normalize(positionGeometry);
    const { pos, nrm, height } = orbDeform(p, mode, uniforms);
    normalLocal.assign(nrm);
    if (mode === DeformMode.Ocean) {
      wave.assign(height);
      obj.assign(p);
    }
    return pos;
  })();

  if (mode === DeformMode.Ocean) {
    const foam = oceanFoam(obj, wave, uniforms);
    material.colorNode = oceanColor(wave, foam, uniforms);
    material.roughnessNode = mix(materialRoughness, 0.75, foam);
    material.emissiveNode = uniforms.uScatter
      .mul(smoothstep(0, 0.05, wave))
      .mul(float(1).sub(foam))
      .mul(0.35);
  }
  return { material, uniforms };
}

/** A dense, indexed icosphere. Only positions matter: the shader rebuilds normals. */
export function createDeformGeometry(coarse: boolean): BufferGeometry {
  const ico = new IcosahedronGeometry(1, coarse ? 44 : 72);
  ico.deleteAttribute('uv');
  return mergeVertices(ico);
}
