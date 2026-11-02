import type { Node } from 'three/webgpu';
import {
  abs,
  clamp,
  float,
  Fn,
  length,
  max,
  min,
  mix,
  normalize,
  pow,
  sign,
  smoothstep,
  vec3,
} from 'three/tsl';
import { morphUniform, planeUniform } from '../scene/shape';
import { UNFURL } from '../core/transition';
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

/** The undeformed ground position in authored detail units. */
export function groundDetail(p: Node<'vec3'>): Node<'vec3'> {
  return planePointNode(p)
    .sub(vec3(0, PLANE.groundY, 1))
    .div(PLANE.detail);
}

const localPlaneWeight = Fn(([p, progress]: [Node<'vec3'>, Node<'float'>]) => {
  const q = planePointNode(p);
  const distance = length(q.xz.sub(vec3(0, 0, PLANE.footZ).xz));
  const delay = smoothstep(PLANE.unfold.near, PLANE.unfold.far, distance).mul(PLANE.unfold.delay);
  return smoothstep(delay, delay.add(PLANE.unfold.duration), progress);
}).setLayout({
  name: 'localPlaneWeight',
  type: 'float',
  inputs: [
    { name: 'p', type: 'vec3' },
    { name: 'progress', type: 'float' },
  ],
});

export function shapePlaneWeight(p: Node<'vec3'>): Node<'float'> {
  return localPlaneWeight(p, planeUniform) as Node<'float'>;
}

/** Keep the advancing rim compact until it unfolds out beyond the camera. */
const unfoldingPoint = Fn(([point, progress, direction]: [Node<'vec3'>, Node<'float'>, Node<'vec3'>]) => {
  const t = smoothstep(0, 1, progress);
  const radius = mix(float(UNFURL.startRadius), float(UNFURL.endRadius), t.mul(t).mul(t));
  const delta = point.xz.sub(vec3(0, 0, 1).xz);
  const scale = min(float(1), radius.div(max(length(delta), 1e-6)));
  const at = delta.mul(scale).add(vec3(0, 0, 1).xz);
  // The back pole is the opening seam: curl its triangles under the visible surface, not through it.
  const back = float(1).sub(smoothstep(UNFURL.undersideUntil, UNFURL.undersideFrom, direction.z));
  const tuck = progress
    .mul(float(1).sub(progress))
    .mul(4 * UNFURL.undersideDepth)
    .mul(back);
  return vec3(at.x, point.y.sub(tuck), at.y);
}).setLayout({
  name: 'unfoldingPoint',
  type: 'vec3',
  inputs: [
    { name: 'point', type: 'vec3' },
    { name: 'progress', type: 'float' },
    { name: 'direction', type: 'vec3' },
  ],
});

/** Open the hidden back cap before its pole-crossing triangles can span the visible ground. */
export function shapeMask(p: Node<'vec3'>): Node<'bool'> {
  const opening = smoothstep(UNFURL.seamFrom, UNFURL.seamUntil, planeUniform);
  return p.z.greaterThan(mix(float(-1.001), float(UNFURL.seamPole), opening));
}

/** A point x built round the orb (in its object space), where the shape draws it. */
export function shapeShell(x: Node<'vec3'>, p: Node<'vec3'> = normalize(x)): Node<'vec3'> {
  return mix(x, unfoldingPoint(planeShellNode(x, baseR(normalize(x))), planeUniform, p), shapePlaneWeight(p));
}

/** A vector v at direction p (a strand's run, a droplet's motion), as the shape turns it. */
export function shapeVector(p: Node<'vec3'>, v: Node<'vec3'>): Node<'vec3'> {
  return mix(v, planeVectorNode(p, v), shapePlaneWeight(p));
}

/** A unit normal n at direction p, as the shape turns it. */
export function shapeNormal(p: Node<'vec3'>, n: Node<'vec3'>): Node<'vec3'> {
  return normalize(mix(n, planeNormalNode(p, n), shapePlaneWeight(p)));
}

/** How much heights off the surface at p are scaled up as drawn: 1 on the orb, more out across the plane. */
export function shapeHeightScale(p: Node<'vec3'>): Node<'float'> {
  return mix(float(1), planeScaleNode(p).y, shapePlaneWeight(p));
}

/**
 * A strand off the surface at direction p, as the shape draws it: on the orb or cube, `orb` (its point there); on the
 * plane, standing on p's ground point with its offset `off` from its root turned onto the ground and drawn at the
 * ground's detail size, the same everywhere, so far strands shrink with perspective as near ones don't grow.
 */
export function shapeStrand(orb: Node<'vec3'>, p: Node<'vec3'>, off: Node<'vec3'>): Node<'vec3'> {
  return mix(
    orb,
    unfoldingPoint(planePointNode(p), planeUniform, p).add(planeRigidNode(p, off).mul(PLANE.detail)),
    shapePlaneWeight(p),
  );
}

/**
 * How far out across the plane p lies, by the chart's scale there against the ground's detail size: 0 near (and on
 * the orb and cube), 1 where strands drawn at the detail size stand several times their spacing apart and cover
 * little of the ground.
 */
export function shapeFar(p: Node<'vec3'>): Node<'float'> {
  return smoothstep(PLANE.detail, PLANE.detail * 4, planeScaleNode(p).x).mul(shapePlaneWeight(p));
}

/** A strand's direction at p, as the shape turns it (shapeStrand). */
export function shapeStrandVector(p: Node<'vec3'>, v: Node<'vec3'>): Node<'vec3'> {
  return mix(v, planeRigidNode(p, v), shapePlaneWeight(p));
}

/** A size the shape scales by `onPlane` on the plane's ground, and leaves alone on the orb and the cube. */
export function shapeOnPlane(onPlane: number, p: Node<'vec3'>): Node<'float'> {
  return mix(float(1), float(onPlane), shapePlaneWeight(p));
}

/**
 * Which way is down for hair hanging off the shape, in world space: the world's down on the orb, and on the plane
 * `inward`, toward the orb's centre from the hair's root, which the plane draws as down.
 */
export function shapeDown(inward: Node<'vec3'>, p: Node<'vec3'>): Node<'vec3'> {
  return normalize(mix(vec3(0, -1, 0), inward, shapePlaneWeight(p)));
}
