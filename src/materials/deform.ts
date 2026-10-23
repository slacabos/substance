import {
  Color,
  IcosahedronGeometry,
  MeshPhysicalMaterial,
  Vector4,
  type BufferGeometry,
  type IUniform,
} from 'three';
import type { MeshPhysicalMaterialParameters } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { MAX_IMPULSES } from '../config';
import { morphUniform } from '../scene/shape';
import deformGlsl from '../shaders/deform.vert.glsl?raw';
import oceanFragGlsl from '../shaders/ocean.frag.glsl?raw';
import shapeGlsl from '../shaders/shape.glsl?raw';
import simplexGlsl from '../shaders/simplex.glsl?raw';

/** Selects the displacement branch in deform.vert.glsl. */
export const DeformMode = { Water: 0, Slime: 1, Ocean: 2 } as const;
export type DeformMode = (typeof DeformMode)[keyof typeof DeformMode];

export interface DeformUniforms {
  [name: string]: IUniform;
  uTime: IUniform<number>;
  uImp: IUniform<Vector4[]>;
  uImpB: IUniform<Vector4[]>;
  uPress: IUniform<Vector4>;
  uWob: IUniform<Vector4>;
  uMorph: IUniform<number>;
  uStorm: IUniform<number>;
  uDeep: IUniform<Color>;
  uShallow: IUniform<Color>;
  uScatter: IUniform<Color>;
}

const vec4s = (n: number) => Array.from({ length: n }, () => new Vector4());

/**
 * A MeshPhysicalMaterial whose vertices are displaced on the GPU by deform.vert.glsl.
 * Normals come from finite differences of the same displacement, so ripples and dents light correctly.
 */
export function createDeformMaterial(
  params: MeshPhysicalMaterialParameters,
  mode: DeformMode,
): { material: MeshPhysicalMaterial; uniforms: DeformUniforms } {
  const material = new MeshPhysicalMaterial(params);
  const uniforms: DeformUniforms = {
    uTime: { value: 0 },
    uImp: { value: vec4s(MAX_IMPULSES).map((v) => v.set(0, 1, 0, 0)) },
    uImpB: { value: vec4s(MAX_IMPULSES) },
    uPress: { value: new Vector4(0, 0, 1, 0) },
    uWob: { value: new Vector4(0, 1, 0, 0) },
    uMorph: morphUniform,
    uStorm: { value: 1 },
    uDeep: { value: new Color('#021a2e') },
    uShallow: { value: new Color('#0b6f85') },
    uScatter: { value: new Color('#1fb5a8') },
  };
  material.defines = { ...material.defines, MODE: mode, MAX_IMPULSES };

  const vertexPrelude = [mode === DeformMode.Slime ? simplexGlsl : '', shapeGlsl, deformGlsl].join('\n');
  const isOcean = mode === DeformMode.Ocean;

  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${vertexPrelude}`)
      .replace(
        '#include <beginnormal_vertex>',
        'vec3 dPos; vec3 dNrm; orbDeform(normalize(position), dPos, dNrm);\nvec3 objectNormal = dNrm;',
      )
      .replace(
        '#include <begin_vertex>',
        `vec3 transformed = dPos;${isOcean ? '\nvWave = orbHeight; vObj = normalize(position);' : ''}`,
      );
    if (isOcean) {
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>\n${oceanFragGlsl}`)
        .replace(
          '#include <color_fragment>',
          '#include <color_fragment>\nfloat foam = oceanFoam();\ndiffuseColor.rgb = oceanColor(foam);',
        )
        .replace(
          '#include <roughnessmap_fragment>',
          '#include <roughnessmap_fragment>\nroughnessFactor = mix(roughnessFactor, 0.75, foam);',
        )
        .replace(
          '#include <emissivemap_fragment>',
          '#include <emissivemap_fragment>\ntotalEmissiveRadiance += uScatter * smoothstep(0.0, 0.05, vWave) * (1.0 - foam) * 0.35;',
        );
    }
  };
  material.customProgramCacheKey = () => `substance-deform-${mode}`;
  return { material, uniforms };
}

/** A dense, indexed icosphere. Only positions matter: the shader rebuilds normals. */
export function createDeformGeometry(coarse: boolean): BufferGeometry {
  const ico = new IcosahedronGeometry(1, coarse ? 44 : 72);
  ico.deleteAttribute('uv');
  return mergeVertices(ico);
}
