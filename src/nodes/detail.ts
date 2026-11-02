import type { Material, Scene } from 'three';
import type { Node, NodeBuilder } from 'three/webgpu';
import { element, float, Fn, If } from 'three/tsl';
import { planeUniform } from '../scene/shape';
import { shapePlaneWeight } from './shape';

/** Uniform branches keep derivatives valid and avoid evaluating a dormant endpoint. */
export function blendDetail<T extends 'vec3' | 'vec4' | 'mat3'>(
  p: Node<'vec3'>,
  zero: Node<T>,
  orb: () => Node<T>,
  ground: () => Node<T>,
): Node<T> {
  return Fn((_inputs: [], builder: NodeBuilder) => {
    // Specialize rendered materials at settled endpoints. Compute kernels keep uniform branching.
    if (builder.material) {
      const mode = builder.material.userData['planeDetailMode'] ?? 0;
      if (mode === 0) return orb();
      if (mode === 1) return ground();
    }
    // Arithmetic is shared by vectors and matrices; TSL's union overloads need a concrete numeric view.
    const result = (zero as unknown as Node<'vec3'>).toVar();
    If(planeUniform.lessThan(1), () => {
      result.assign(orb() as unknown as Node<'vec3'>);
    });
    If(planeUniform.greaterThan(0), () => {
      const w = shapePlaneWeight(p);
      result.assign(result.mul(float(1).sub(w)).add((ground() as unknown as Node<'vec3'>).mul(w)));
    });
    return result;
  })() as unknown as Node<T>;
}

/** WGSL matrix indexing returns a column; the current Three typings only expose array indexing. */
export function detailColumn(matrix: Node<'mat3'>, index: number): Node<'vec3'> {
  return element(matrix as unknown as Parameters<typeof element>[0], index) as unknown as Node<'vec3'>;
}

const keyed = new WeakSet<Material>();

/** Change shader variants only at endpoint/transition boundaries, never on every morph frame. */
export function setSceneDetailMode(scene: Scene, mode: number): void {
  scene.traverse((object) => {
    if (!('material' in object)) return;
    const materials = (Array.isArray(object.material) ? object.material : [object.material]) as Material[];
    for (const material of materials) {
      if (material.userData['planeDetailMode'] === mode) continue;
      if (!keyed.has(material)) {
        const key = material.customProgramCacheKey.bind(material);
        material.customProgramCacheKey = () =>
          `${key()}|planeDetail:${material.userData['planeDetailMode'] ?? 0}`;
        keyed.add(material);
      }
      material.userData['planeDetailMode'] = mode;
      material.needsUpdate = true;
    }
  });
}
