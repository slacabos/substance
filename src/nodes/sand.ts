import type { Node } from 'three/webgpu';
import {
  abs,
  color,
  cross,
  dot,
  float,
  fwidth,
  length,
  max,
  mix,
  normalize,
  normalView,
  positionView,
  sign,
  sin,
  smoothstep,
  vec3,
} from 'three/tsl';
import { snoise } from './noise';

type Float = Node<'float'>;
type Vec3 = Node<'vec3'>;

// Sand up close, which no mesh is fine enough to carry: grain, wind ripples and their bump, shared by the coast's
// beaches (nodes/shore.ts) and the sand orb (materials/sand.ts). Detail finer than a couple of pixels fades out,
// so nothing shimmers.

/** How far the bump raises and lowers the surface, orb radii. */
export const BUMP = 0.0012;

/** Simplex noise at frequency f, fading to nothing once its features shrink below a couple of pixels. */
export function octave(p: Vec3, f: number): Float {
  const at = p.mul(f);
  const pixels = length(fwidth(at));
  return snoise(at).mul(float(1).sub(smoothstep(0.35, 0.7, pixels)));
}

/**
 * Parallel bands across p, `f` to the radian along `axis`, bent by `warp`: 0 to 1, flattening to 0.5 once they
 * get finer than a couple of pixels.
 */
export function bands(p: Vec3, axis: Vec3, f: number, warp: Float): Float {
  const phase = dot(p, axis).mul(f).add(warp);
  const pixels = fwidth(phase).div(Math.PI * 2);
  return sin(phase)
    .mul(float(1).sub(smoothstep(0.25, 0.5, pixels)))
    .mul(0.5)
    .add(0.5);
}

/** A normal tilted by a height field `h` (orb radii) from its screen-space derivatives (Mikkelsen 2010). */
export function bumped(h: Float): Vec3 {
  const dpx = positionView.dFdx();
  const dpy = positionView.dFdy();
  const n = normalView;
  const r1 = cross(dpy, n);
  const r2 = cross(n, dpx);
  const det = dot(dpx, r1);
  const grad = r1.mul(h.dFdx()).add(r2.mul(h.dFdy())).mul(sign(det));
  return normalize(n.mul(abs(det)).sub(grad));
}

/** The colours a sand orb is made of: dry, a warmer and a paler drift, and damp. */
const DRY = color('#b89a6a');
const WARM = color('#c99d6a');
const PALE = color('#ddd0b0');
const WET = color('#7d6a4b');

/**
 * Sand's grain and wind ripples at unit direction p: a tone (about 1) and a bump height (orb radii). `combed` (0 to
 * 1) is how much of the wind's ripple pattern is left: freshly disturbed sand has none until the wind combs it
 * again.
 */
export function sandGrain(p: Vec3, combed: Float = float(1)): { tone: Float; height: Float } {
  const grain = octave(p, 900);
  const ripple = bands(p, vec3(0.8, 0.1, -0.59), 1400, octave(p, 60).mul(4));
  return {
    tone: grain.mul(0.06).add(0.97).sub(ripple.mul(0.04).mul(combed)),
    height: grain.mul(0.25).add(ripple.mul(0.35).mul(combed)).mul(BUMP),
  };
}

/**
 * A whole orb of sand at unit direction p: drifts of warmer and paler sand; `wet` (0 to 1) darkens and glosses
 * it; `disturbed` (1 freshly moved, 0 settled) is the damper sand from underneath, darker, lumpier, with the
 * ripples wiped out; `below` (orb radii under the rest surface) damper still, the deeper it's dug.
 */
export function sandSurface(
  p: Vec3,
  wet: Float,
  disturbed: Float,
  below: Float = float(0),
): { color: Vec3; roughness: Float; normal: Vec3 } {
  const drift = octave(p, 3.5).add(octave(p, 9).mul(0.3));
  const lumps = octave(p, 520)
    .mul(disturbed)
    .mul(BUMP * 0.8);
  const { tone, height } = sandGrain(p, float(1).sub(disturbed));
  const base = mix(
    mix(DRY, WARM, smoothstep(0, 0.8, drift).mul(0.25)),
    PALE,
    smoothstep(0, 0.8, drift.negate()).mul(0.18),
  );
  // sand turned up from underneath is damp, and dries back as it settles
  const damp = max(wet, max(disturbed.mul(0.45), smoothstep(0, 0.02, below).mul(0.55)));
  return {
    color: mix(base, WET, damp).mul(tone),
    roughness: mix(mix(float(0.9), float(0.96), disturbed), float(0.45), wet),
    normal: bumped(height.add(lumps).mul(float(1).sub(wet.mul(0.6)))),
  };
}
