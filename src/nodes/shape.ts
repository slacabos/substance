import type { Node } from 'three/webgpu';
import { abs, clamp, float, mix, normalize, pow, sign, smoothstep, vec3 } from 'three/tsl';
import { morphUniform, planeUniform } from '../scene/shape';
import { PLANE } from '../core/chart';
import {
  planeNormalNode,
  planePointNode,
  planeRigidNode,
  planeScaleNode,
  planeShellNode,
  planeVectorNode,
} from './chart';

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

/**
 * A strand off the surface at direction p, as the shape draws it: on the orb or cube, `orb` (its point there); on the
 * plane, standing on p's ground point with its offset `off` from its root turned onto the ground and drawn at the
 * ground's detail size, the same everywhere, so far strands shrink with perspective as near ones don't grow.
 */
export function shapeStrand(orb: Node<'vec3'>, p: Node<'vec3'>, off: Node<'vec3'>): Node<'vec3'> {
  return mix(orb, planePointNode(p).add(planeRigidNode(p, off).mul(PLANE.detail)), planeUniform);
}

/**
 * How far out across the plane p lies, by the chart's scale there against the ground's detail size: 0 near (and on
 * the orb and cube), 1 where strands drawn at the detail size stand several times their spacing apart and cover
 * little of the ground.
 */
export function shapeFar(p: Node<'vec3'>): Node<'float'> {
  return smoothstep(PLANE.detail, PLANE.detail * 4, planeScaleNode(p).x).mul(planeUniform);
}

/** A strand's direction at p, as the shape turns it (shapeStrand). */
export function shapeStrandVector(p: Node<'vec3'>, v: Node<'vec3'>): Node<'vec3'> {
  return mix(v, planeRigidNode(p, v), planeUniform);
}

/** A size the shape scales by `onPlane` on the plane's ground, and leaves alone on the orb and the cube. */
export function shapeOnPlane(onPlane: number): Node<'float'> {
  return mix(float(1), float(onPlane), planeUniform);
}

/**
 * Which way is down for hair hanging off the shape, in world space: the world's down on the orb, and on the plane
 * `inward`, toward the orb's centre from the hair's root, which the plane draws as down.
 */
export function shapeDown(inward: Node<'vec3'>): Node<'vec3'> {
  return normalize(mix(vec3(0, -1, 0), inward, planeUniform));
}
