import { Vector3, type Color, type Vector4 } from 'three';
import type { Node, UniformArrayNode, UniformNode } from 'three/webgpu';
import {
  abs,
  acos,
  clamp,
  Continue,
  cos,
  cross,
  dot,
  exp,
  float,
  If,
  Loop,
  max,
  normalize,
  select,
  sin,
  smoothstep,
  vec3,
} from 'three/tsl';
import { MAX_IMPULSES } from '../config';
import { snoise } from './noise';
import { baseR } from './shape';

// GPU displacement for the water-like materials. Every surface point is a unit direction pushed out by
// baseR (the orb/cube shape) plus a height from ripples, dents, swells or wobble.

/** Which displacement a deform material uses. */
export const DeformMode = { Water: 0, Slime: 1, Ocean: 2 } as const;
export type DeformMode = (typeof DeformMode)[keyof typeof DeformMode];

export interface DeformUniforms {
  uTime: UniformNode<'float', number>;
  /** Water/ocean: xyz direction, w start time. Slime: xyz direction, w depth. */
  uImp: UniformArrayNode<'vec4'>;
  /** Water/ocean: x strength. Slime: x radius. */
  uImpB: UniformArrayNode<'vec4'>;
  /** The arrays behind uImp and uImpB: write to these, they upload every frame. */
  imp: Vector4[];
  impB: Vector4[];
  /** xyz direction, w how hard the finger is pressing. */
  uPress: UniformNode<'vec4', Vector4>;
  /** xyz axis, w amount of the squash-and-stretch mode. */
  uWob: UniformNode<'vec4', Vector4>;
  uStorm: UniformNode<'float', number>;
  uDeep: UniformNode<'color', Color>;
  uShallow: UniformNode<'color', Color>;
  uScatter: UniformNode<'color', Color>;
}

type Vec3 = Node<'vec3'>;
type Float = Node<'float'>;

const arcDist = (a: Vec3, b: Vec3) => acos(clamp(dot(a, b), -1, 1));
const unit = (x: number, y: number, z: number) => vec3(new Vector3(x, y, z).normalize());

/** Sharp-crested swell: exp(sin) narrows the peaks and flattens the troughs, like wind waves. */
const crest = (x: Float) => exp(sin(x).sub(1)).sub(0.466);

// amplitude, direction, frequency, warp factor, speed
const SWELLS: [number, [number, number, number], number, number, number][] = [
  [0.02, [1, 0.25, 0.35], 6, 1, 1.05],
  [0.015, [0.6, -0.3, 1], 8.5, -1, 1.3],
  [0.011, [-0.4, 0.8, 0.5], 12, 0.5, 1.6],
  [0.01, [0.9, 0.5, -0.6], 18, 0, 2],
  [0.006, [-0.7, -0.2, 0.8], 27, 0, 2.5],
  [0.0035, [0.2, -0.9, -0.4], 39, 0, 3.1],
  [0.0025, [-0.5, 0.4, -0.8], 47, 1, 3.4],
];

function oceanAt(p: Vec3, u: DeformUniforms): Float {
  const t = u.uTime;
  const warp = sin(
    dot(p, vec3(0.3, 0.9, 0.2))
      .mul(3)
      .add(t.mul(0.25)),
  ).mul(0.5);
  let d: Float = float(0);
  for (const [amp, [x, y, z], freq, warpK, speed] of SWELLS) {
    const phase = dot(p, unit(x, y, z))
      .mul(freq)
      .add(warp.mul(warpK))
      .sub(t.mul(speed));
    d = d.add(crest(phase).mul(amp));
  }
  return d.mul(u.uStorm);
}

/** Expanding, decaying rings from each tap or drag step, plus the dip under a pressing finger. */
function ripplesAt(p: Vec3, u: DeformUniforms, d: Float): void {
  const t = u.uTime;
  Loop(MAX_IMPULSES, ({ i }) => {
    const a = u.uImp.element(i);
    const str = u.uImpB.element(i).x;
    const age = t.sub(a.w);
    If(str.lessThanEqual(0).or(age.lessThan(0)).or(age.greaterThan(5)), () => {
      Continue();
    });
    const s = arcDist(p, a.xyz);
    const behind = age.mul(0.85).sub(s); // distance behind the wavefront
    const env = smoothstep(-0.06, 0.04, behind)
      .mul(exp(max(behind, 0).mul(-3.2)))
      .mul(exp(age.mul(-0.75)))
      .div(s.mul(1.5).add(1));
    d.addAssign(
      str
        .mul(0.045)
        .mul(env)
        .mul(cos(behind.mul(17))),
    );
    d.subAssign(
      str
        .mul(0.05)
        .mul(exp(s.mul(s).mul(-90)))
        .mul(exp(age.mul(-5))),
    ); // the initial splash dip
  });
  const ps = arcDist(p, u.uPress.xyz);
  d.subAssign(u.uPress.w.mul(0.035).mul(exp(ps.mul(ps).mul(-70))));
}

/** Slow noise, spring-driven dents (each shoving goo aside) and a squash-and-stretch mode. */
function slimeAt(p: Vec3, u: DeformUniforms, d: Float): void {
  const t = u.uTime;
  d.addAssign(snoise(p.mul(2.1).add(vec3(0, t.mul(0.18), t.mul(0.07)))).mul(0.01));
  d.addAssign(snoise(p.mul(5.3).sub(vec3(t.mul(0.12)))).mul(0.003));
  Loop(MAX_IMPULSES, ({ i }) => {
    const a = u.uImp.element(i);
    If(abs(a.w).lessThan(1e-4), () => {
      Continue();
    });
    const r = u.uImpB.element(i).x;
    const s = arcDist(p, a.xyz);
    const rim = s.sub(r.mul(1.7)).div(r.mul(0.75));
    d.addAssign(a.w.mul(exp(s.mul(s).div(r.mul(r)).negate()).sub(exp(rim.mul(rim).negate()).mul(0.32))));
  });
  const c = dot(p, u.uWob.xyz);
  d.addAssign(u.uWob.w.mul(c.mul(c).mul(1.5).sub(0.5)));
}

function dispAt(p: Vec3, mode: DeformMode, u: DeformUniforms): Float {
  const d = float(0).toVar();
  if (mode === DeformMode.Slime) {
    slimeAt(p, u, d);
    return d;
  }
  if (mode === DeformMode.Water) {
    // gentle swell so the water never looks frozen
    const t = u.uTime;
    d.addAssign(
      sin(p.x.mul(5).add(t.mul(1.3)))
        .mul(sin(p.y.mul(4).sub(t.mul(1.1))))
        .mul(0.006),
    );
    d.addAssign(sin(p.z.mul(7).add(p.x.mul(3)).add(t.mul(1.7))).mul(0.004));
  } else {
    d.addAssign(oceanAt(p, u));
  }
  ripplesAt(p, u, d);
  return d;
}

/**
 * Displaced position plus a finite-difference normal, so any displacement shades correctly. Must be called
 * inside a TSL Fn (it emits loops).
 */
export function orbDeform(
  p: Vec3,
  mode: DeformMode,
  u: DeformUniforms,
): { pos: Vec3; nrm: Vec3; height: Float } {
  const height = dispAt(p, mode, u);
  const pos = p.mul(baseR(p).add(height)).toVar();
  const t1 = normalize(cross(p, select(abs(p.y).lessThan(0.99), vec3(0, 1, 0), vec3(1, 0, 0))));
  const t2 = cross(p, t1);
  const e = 0.004;
  const pa = normalize(p.add(t1.mul(e)));
  const pb = normalize(p.add(t2.mul(e)));
  const A = pa.mul(baseR(pa).add(dispAt(pa, mode, u)));
  const B = pb.mul(baseR(pb).add(dispAt(pb, mode, u)));
  const n = normalize(cross(A.sub(pos), B.sub(pos)));
  const nrm = select(dot(n, p).lessThan(0), n.negate(), n);
  return { pos, nrm, height };
}
