import type { Node } from 'three/webgpu';
import { abs, clamp, float, mix, normalize, pow, sign, vec3 } from 'three/tsl';
import { PLANE } from '../core/chart';
import { morphUniform, planeUniform } from '../scene/shape';
import { planeNormalNode, planeScaleNode, planeShellNode, planeVectorNode } from './chart';

// Sphere-to-rounded-cube morph. Every surface point is a unit direction p scaled by baseR(p);
// at morph = 1 that traces the superellipsoid |x|^10 + |y|^10 + |z|^10 = 0.8^10.
// The plane lays that out flat on top (nodes/chart.ts): whatever's built round the orb is drawn through
// shapeShell, its normals through shapeNormal.

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

/** A point x built round the orb (in its object space), where the shape draws it. */
export function shapeShell(x: Node<'vec3'>): Node<'vec3'> {
  return mix(x, planeShellNode(x, baseR(normalize(x))), planeUniform);
}

/** A vector v at direction p (a strand's run, a droplet's motion), as the shape turns it. */
export function shapeVector(p: Node<'vec3'>, v: Node<'vec3'>): Node<'vec3'> {
  return mix(v, planeVectorNode(p, v), planeUniform);
}

/** A unit normal n at direction p, as the shape turns it. */
export function shapeNormal(p: Node<'vec3'>, n: Node<'vec3'>): Node<'vec3'> {
  return normalize(mix(n, planeNormalNode(p, n), planeUniform));
}

/** How much heights off the surface at p are scaled up as drawn: 1 on the orb, more out across the plane. */
export function shapeHeightScale(p: Node<'vec3'>): Node<'float'> {
  return mix(float(1), planeScaleNode(p).y, planeUniform);
}

/** How long hair is drawn, against the orb's: shorter on the plane (PLANE.hair), and thinner with it. */
export function shapeHair(): Node<'float'> {
  return mix(float(1), float(PLANE.hair), planeUniform);
}

/**
 * Which way is down for hair hanging off the shape, in world space: the world's down on the orb, and on the plane
 * `inward`, toward the orb's centre from the hair's root, which the plane draws as down.
 */
export function shapeDown(inward: Node<'vec3'>): Node<'vec3'> {
  return normalize(mix(vec3(0, -1, 0), inward, planeUniform));
}
