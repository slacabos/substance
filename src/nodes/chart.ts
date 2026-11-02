import type { Node } from 'three/webgpu';
import {
  abs,
  clamp,
  cos,
  cross,
  dot,
  exp,
  float,
  Fn,
  length,
  max,
  min,
  normalize,
  select,
  sin,
  vec3,
  vec4,
} from 'three/tsl';
import { PLANE } from '../core/chart';

// The plane chart (core/chart.ts) on the GPU, line for line. Each is one WGSL function that every use calls, as
// nodes/noise.ts is, so the shaders that lay several things out flat don't each copy its locals in again (Safari's
// WebGPU allows a shader 8 KB of them).

type Vec3 = Node<'vec3'>;

const K = PLANE.kappa;
const TURN_COS = Math.cos(PLANE.turn);
const TURN_SIN = Math.sin(PLANE.turn);

/** The orb turned about the axis toward the camera (the chart's step 0). */
const turned = (v: Vec3): Vec3 =>
  vec3(v.x.mul(TURN_COS).sub(v.y.mul(TURN_SIN)), v.x.mul(TURN_SIN).add(v.y.mul(TURN_COS)), v.z);

/** The fan at direction p: (ρ, φ, σx, σy). */
const chartFan = Fn(([direction]: [Vec3]) => {
  const p = turned(direction);
  const d = max(p.z.add(1), 1e-6);
  const sx = p.x.mul(2).div(d);
  const sy = p.y.mul(2).div(d);
  const rho = clamp(exp(sy.mul(-K)).mul(PLANE.rho0), PLANE.rhoMin, PLANE.rhoFar);
  const phi = clamp(sx.mul(-K), -PLANE.phiMax, PLANE.phiMax);
  return vec4(rho, phi, sx, sy);
}).setLayout({ name: 'chartFan', type: 'vec4', inputs: [{ name: 'p', type: 'vec3' }] });

/** Where the direction p lies on the ground. */
const chartPoint = Fn(([p]: [Vec3]) => {
  const fan = chartFan(p);
  return vec3(fan.x.mul(sin(fan.y)), PLANE.groundY, float(PLANE.footZ).sub(fan.x.mul(cos(fan.y))));
}).setLayout({ name: 'chartPoint', type: 'vec3', inputs: [{ name: 'p', type: 'vec3' }] });

/** Ground length per unit of direction around p, and the same capped for heights: (s, h). */
const chartScale = Fn(([p]: [Vec3]) => {
  const fan = chartFan(p);
  const s = fan.x.mul(fan.z.mul(fan.z).add(fan.w.mul(fan.w)).add(4)).mul(K / 4);
  return vec4(s, min(s, PLANE.heightCap), 0, 0);
}).setLayout({ name: 'chartScale', type: 'vec4', inputs: [{ name: 'p', type: 'vec3' }] });

/** Where a step t along the orb's surface at p goes on the ground. */
const chartPush = Fn(([direction, step]: [Vec3, Vec3]) => {
  const fan = chartFan(direction);
  const p = turned(direction);
  const t = turned(step);
  const d = max(p.z.add(1), 1e-6);
  const dd = d.mul(d);
  const dsx = t.x.mul(d).sub(p.x.mul(t.z)).mul(2).div(dd);
  const dsy = t.y.mul(d).sub(p.y.mul(t.z)).mul(2).div(dd);
  const dRho = fan.x.mul(dsy).mul(-K);
  const dPhi = dsx.mul(-K);
  const s = sin(fan.y);
  const c = cos(fan.y);
  return vec3(dRho.mul(s).add(fan.x.mul(dPhi).mul(c)), 0, dRho.negate().mul(c).add(fan.x.mul(dPhi).mul(s)));
}).setLayout({
  name: 'chartPush',
  type: 'vec3',
  inputs: [
    { name: 'p', type: 'vec3' },
    { name: 't', type: 'vec3' },
  ],
});

/**
 * The step along the orb's surface at p that the chart carries along the ground vector g (its scale aside): the
 * transpose of its differential, which for a conformal map points the same way as its inverse.
 */
const chartPull = Fn(([p, g]: [Vec3, Vec3]) => {
  const e1 = normalize(cross(p, select(abs(p.y).lessThan(0.9), vec3(0, 1, 0), vec3(1, 0, 0))));
  const e2 = cross(p, e1);
  return e1.mul(dot(g, chartPush(p, e1))).add(e2.mul(dot(g, chartPush(p, e2))));
}).setLayout({
  name: 'chartPull',
  type: 'vec3',
  inputs: [
    { name: 'p', type: 'vec3' },
    { name: 'g', type: 'vec3' },
  ],
});

/** A point x near the surface (its base surface `base` out along its direction) moved onto the ground. */
const chartShell = Fn(([x, base]: [Vec3, Node<'float'>]) => {
  const p = normalize(x);
  const ground = chartPoint(p);
  return vec3(ground.x, length(x).sub(base).mul(chartScale(p).y).add(PLANE.groundY), ground.z);
}).setLayout({
  name: 'chartShell',
  type: 'vec3',
  inputs: [
    { name: 'x', type: 'vec3' },
    { name: 'base', type: 'float' },
  ],
});

/** A vector v at direction p (along the surface and off it) as the shell map turns it. */
const chartVector = Fn(([p, v]: [Vec3, Vec3]) => {
  const out = v.dot(p);
  return chartPush(p, v.sub(p.mul(out))).add(vec3(0, out.mul(chartScale(p).y), 0));
}).setLayout({
  name: 'chartVector',
  type: 'vec3',
  inputs: [
    { name: 'p', type: 'vec3' },
    { name: 'v', type: 'vec3' },
  ],
});

/** A vector v at direction p turned as the shell map turns it, but not scaled: the chart's rotation alone. */
const chartRigid = Fn(([p, v]: [Vec3, Vec3]) => {
  const out = v.dot(p);
  return chartPush(p, v.sub(p.mul(out)))
    .div(chartScale(p).x)
    .add(vec3(0, out, 0));
}).setLayout({
  name: 'chartRigid',
  type: 'vec3',
  inputs: [
    { name: 'p', type: 'vec3' },
    { name: 'v', type: 'vec3' },
  ],
});

/** A surface normal n at direction p as the shell map turns it (the inverse transpose of its Jacobian). */
const chartNormal = Fn(([p, n]: [Vec3, Vec3]) => {
  const scale = chartScale(p);
  const along = n.dot(p);
  const turned = chartPush(p, n.sub(p.mul(along))).mul(scale.y.div(scale.x.mul(scale.x)));
  return normalize(turned.add(vec3(0, along, 0)));
}).setLayout({
  name: 'chartNormal',
  type: 'vec3',
  inputs: [
    { name: 'p', type: 'vec3' },
    { name: 'n', type: 'vec3' },
  ],
});

export const planePointNode = (p: Vec3): Vec3 => chartPoint(p) as Vec3;
/** (s, h): the ground's scale at p, and the same capped for heights. */
export const planeScaleNode = (p: Vec3): Node<'vec4'> => chartScale(p) as Node<'vec4'>;
export const planeShellNode = (x: Vec3, base: Node<'float'>): Vec3 => chartShell(x, base) as Vec3;
export const planeVectorNode = (p: Vec3, v: Vec3): Vec3 => chartVector(p, v) as Vec3;
/** A vector v at direction p turned onto the ground without being scaled. */
export const planeRigidNode = (p: Vec3, v: Vec3): Vec3 => chartRigid(p, v) as Vec3;
/** The way along the orb's surface at p that comes out along the ground vector g (unnormalized). */
export const planePullNode = (p: Vec3, g: Vec3): Vec3 => chartPull(p, g) as Vec3;
export const planeNormalNode = (p: Vec3, n: Vec3): Vec3 => chartNormal(p, n) as Vec3;
