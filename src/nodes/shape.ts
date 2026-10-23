import type { Node } from 'three/webgpu';
import { abs, clamp, float, mix, normalize, pow, sign, vec3 } from 'three/tsl';
import { morphUniform } from '../scene/shape';

// Sphere-to-rounded-cube morph. Every surface point is a unit direction p scaled by baseR(p);
// at morph = 1 that traces the superellipsoid |x|^10 + |y|^10 + |z|^10 = 0.8^10.

export function baseR(p: Node<'vec3'>): Node<'float'> {
  const a = abs(p);
  const s = pow(pow(a.x, 10).add(pow(a.y, 10)).add(pow(a.z, 10)), 0.1);
  return mix(float(1), float(0.8).div(s), morphUniform);
}

/** Surface normal at q = p * baseR(p), blended so it stays smooth mid-morph. */
export function baseN(p: Node<'vec3'>, q: Node<'vec3'>): Node<'vec3'> {
  const g = sign(q).mul(pow(abs(q), vec3(9)));
  return normalize(mix(p, normalize(g), clamp(morphUniform, 0, 1)));
}
