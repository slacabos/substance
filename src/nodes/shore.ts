import type { Node } from 'three/webgpu';
import {
  abs,
  attribute,
  clamp,
  cross,
  dot,
  float,
  fwidth,
  length,
  mix,
  normalize,
  normalView,
  positionGeometry,
  positionView,
  sign,
  sin,
  smoothstep,
  vec3,
} from 'three/tsl';
import { snoise } from './noise';

type Float = Node<'float'>;
type Vec3 = Node<'vec3'>;

// Close-up detail for the islands (materials/oceanShores.ts), which their vertices are far too coarse to carry:
// each surface gets its own procedural texture and bump, weighted by the per-vertex `terrain` attribute (rock,
// sand, grass, and how wet the sand is at the waterline).

/** How far the bump raises and lowers the surface, orb radii. */
const BUMP = 0.0012;

/** Simplex noise at frequency f, fading to nothing once its features shrink below a couple of pixels. */
function octave(p: Vec3, f: number): Float {
  const at = p.mul(f);
  const pixels = length(fwidth(at));
  return snoise(at).mul(float(1).sub(smoothstep(0.35, 0.7, pixels)));
}

/**
 * Parallel bands across p, `f` to the radian along `axis`, bent by `warp`: 0 to 1, flattening to 0.5 once they
 * get finer than a couple of pixels.
 */
function bands(p: Vec3, axis: Vec3, f: number, warp: Float): Float {
  const phase = dot(p, axis).mul(f).add(warp);
  const pixels = fwidth(phase).div(Math.PI * 2);
  return sin(phase)
    .mul(float(1).sub(smoothstep(0.25, 0.5, pixels)))
    .mul(0.5)
    .add(0.5);
}

/** A normal tilted by a height field `h` (orb radii) from its screen-space derivatives (Mikkelsen 2010). */
function bumped(h: Float): Vec3 {
  const dpx = positionView.dFdx();
  const dpy = positionView.dFdy();
  const n = normalView;
  const r1 = cross(dpy, n);
  const r2 = cross(n, dpx);
  const det = dot(dpx, r1);
  const grad = r1.mul(h.dFdx()).add(r2.mul(h.dFdy())).mul(sign(det));
  return normalize(n.mul(abs(det)).sub(grad));
}

/** The islands' surface: a tone multiplying their vertex colour, roughness, and a bumped view-space normal. */
export function shoreSurface(): { tone: Vec3; roughness: Float; normal: Vec3 } {
  const terrain = attribute('terrain', 'vec4');
  const [rock, sand, grass, wet] = [terrain.x, terrain.y, terrain.z, terrain.w];
  const p = normalize(positionGeometry);

  // rock: broad blotches, layered streaks and dark crevices
  const broad = octave(p, 40).mul(0.5).add(octave(p, 130).mul(0.3)).add(octave(p, 380).mul(0.2));
  const strata = bands(p, vec3(0.2, 0.95, 0.24), 900, octave(p, 30).mul(5));
  const crevice = smoothstep(-0.55, -0.15, broad);
  const rockTone = broad.mul(0.22).add(0.86).add(strata.mul(0.08)).mul(crevice.mul(0.45).add(0.55));
  const rockHeight = broad.add(strata.mul(0.25));

  // sand: fine grain, and faint ripples the wind combed across it
  const grain = octave(p, 900);
  const ripple = bands(p, vec3(0.8, 0.1, -0.59), 1400, octave(p, 60).mul(4));
  const sandTone = grain.mul(0.06).add(0.97).sub(ripple.mul(0.04));
  const sandHeight = grain.mul(0.25).add(ripple.mul(0.35));

  // grass: patchy, with tufts
  const patches = octave(p, 35);
  const tufts = octave(p, 260).mul(0.6).add(octave(p, 700).mul(0.4));
  const grassTone = patches.mul(0.18).add(0.9).add(tufts.mul(0.12));

  const total = clamp(rock.add(sand).add(grass), 0.001, 3);
  const tone = rockTone.mul(rock).add(sandTone.mul(sand)).add(grassTone.mul(grass)).div(total);
  const height = rockHeight
    .mul(rock)
    .add(sandHeight.mul(sand).mul(float(1).sub(wet.mul(0.7))))
    .add(tufts.mul(grass).mul(0.5))
    .div(total)
    .mul(BUMP);
  // wet sand at the waterline is darker and glossy
  return {
    tone: vec3(tone.mul(mix(float(1), float(0.72), wet))),
    roughness: mix(mix(float(0.95), float(0.85), sand), float(0.35), wet),
    normal: bumped(height),
  };
}
