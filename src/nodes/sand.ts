import { Vector3 } from 'three';
import type { Node } from 'three/webgpu';
import {
  abs,
  cameraViewMatrix,
  clamp,
  color,
  cross,
  dot,
  exp2,
  float,
  floor,
  fract,
  fwidth,
  hash,
  ivec3,
  length,
  log2,
  max,
  mix,
  modelViewMatrix,
  normalize,
  normalView,
  normalViewGeometry,
  positionView,
  select,
  sign,
  smoothstep,
  uint,
  vec3,
  vec4,
} from 'three/tsl';
import { snoise } from './noise';

type Float = Node<'float'>;
type Vec3 = Node<'vec3'>;

// Sand up close, which no mesh is fine enough to carry: grains, wind ripples, crumbs and glints, shared by the
// coast's beaches (nodes/shore.ts) and the sand orb (materials/sand.ts). Smooth detail finer than a couple of
// pixels fades out, so nothing shimmers; the grains themselves are drawn at about a pixel and a half whatever the
// orb's size, as sand seen from afar is still a speckle of grains, never a smooth paste.

/** How far the bump raises and lowers the surface, orb radii. */
export const BUMP = 0.0012;

/** Where the sand's key light comes from (world space): upper left front, raking across the near side. */
export const SAND_KEY_LIGHT = new Vector3(-0.55, 0.65, 0.52).normalize();

/** The light sand glints in: world direction toward it, and its colour times strength. */
export interface SandLight {
  dir: Vec3;
  color: Vec3;
}

/** A grain, orb radii: 0.3 mm on an orb about 10 cm across. */
const GRAIN = 0.006;
/** How many pixels across a grain cell is drawn. */
const CELL_PIXELS = 1.5;
/** Share of grains with a facet that can catch the light, and how closely it must face the light to glint. */
const GLINT_SHARE = 0.14;
const GLINT_ALIGN = 0.99;
const GLINT_STRENGTH = 2.2;

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
  return phase
    .sin()
    .mul(float(1).sub(smoothstep(0.25, 0.5, pixels)))
    .mul(0.5)
    .add(0.5);
}

/**
 * Wind ripples across p, `f` to the radian along `axis`: a long gentle slope up the windward side and a short
 * steep lee, 0 at a trough and 1 at a crest, flattening to 0.5 once they get finer than a couple of pixels.
 */
function ripples(p: Vec3, axis: Vec3, f: number, warp: Float): Float {
  const phase = dot(p, axis).mul(f).add(warp);
  const s = fract(phase.div(Math.PI * 2));
  const ramp = select(s.lessThan(0.7), s.div(0.7), float(1).sub(s).div(0.3));
  const shaped = ramp.mul(ramp).mul(float(3).sub(ramp.mul(2)));
  const pixels = fwidth(phase).div(Math.PI * 2);
  return mix(shaped, float(0.5), smoothstep(0.2, 0.45, pixels));
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

/** One grain cell of a lattice at frequency f: its mineral's tint, a random tilt, and whether it can glint. */
function grainCell(p: Vec3, f: Float) {
  const c = ivec3(floor(p.mul(f)));
  const seed = uint(c.x).mul(73856093).bitXor(uint(c.y).mul(19349663)).bitXor(uint(c.z).mul(83492791)).mul(8);
  const r = (j: number) => hash(seed.add(j));
  const kind = r(0);
  // mostly quartz and feldspar, tan with a spread of shade and hue; a few dark grains (magnetite, basalt), a
  // few clear white ones and a few rusty
  const tan = mix(vec3(1.05, 0.98, 0.9), vec3(0.94, 0.99, 1.06), r(1)).mul(r(2).mul(0.36).add(0.82));
  const tint = select(
    kind.lessThan(0.07),
    vec3(0.34, 0.3, 0.27),
    select(
      kind.lessThan(0.12),
      vec3(1.24, 1.22, 1.16),
      select(kind.lessThan(0.16), vec3(1.12, 0.84, 0.7), tan),
    ),
  );
  const tilt = vec3(r(3), r(4), r(5)).mul(2).sub(1);
  return { tint, tilt, glints: r(6).lessThan(GLINT_SHARE) };
}

/**
 * The grains at p, about a pixel and a half each: two lattice levels around that size, blended, so they never
 * pop as the orb grows or shrinks on screen. A cell bigger than a grain stands for several, whose colours average
 * out, so its speckle is softer; it never fades away entirely. Returns the tint to multiply the sand's colour
 * by, the normal tilted grain by grain (n, view space), and the glints of grains facing `light` (emissive).
 */
export function sandGrains(
  p: Vec3,
  n: Vec3,
  light: SandLight,
  wet: Float,
): { tint: Vec3; normal: Vec3; glint: Vec3 } {
  const footprint = max(length(fwidth(p)), 1e-6).mul(CELL_PIXELS);
  const level = log2(footprint).negate();
  const lo = floor(level);
  const t = level.sub(lo);
  const contrast = clamp(float(GRAIN).div(footprint), 0.4, 1);
  const l = normalize(cameraViewMatrix.mul(vec4(light.dir, 0)).xyz);
  const h = normalize(l.add(normalize(positionView).negate()));
  // the glints are emissive, built outside the material's normal pass, where normalView is already the finished
  // normal (this one): so they tilt the surface's own normal instead
  const surface = normalViewGeometry;
  const lit = smoothstep(0, 0.15, dot(surface, l));
  const at = (f: Float) => {
    const cell = grainCell(p, f);
    // the tilt, flat to the surface and turning with the orb
    const tilt = modelViewMatrix.mul(vec4(cell.tilt.sub(p.mul(dot(cell.tilt, p))), 0)).xyz;
    const facet = normalize(surface.add(tilt.mul(0.9)));
    const glint = select(cell.glints, smoothstep(GLINT_ALIGN, 1, dot(facet, h)), float(0));
    return { tint: mix(vec3(1), cell.tint, contrast), tilt, glint };
  };
  const a = at(exp2(lo));
  const b = at(exp2(lo.add(1)));
  const tilt = mix(a.tilt, b.tilt, t);
  // wet sand's film of water smooths the grains over and dulls their glints
  const dry = float(1).sub(wet.mul(0.75));
  return {
    tint: mix(mix(a.tint, b.tint, t), vec3(1), wet.mul(0.45)),
    normal: normalize(n.add(tilt.mul(contrast.mul(0.32).mul(dry)))),
    glint: light.color.mul(mix(a.glint, b.glint, t).mul(lit).mul(dry).mul(GLINT_STRENGTH)),
  };
}

/** Where sand stands steep (dug walls, pile flanks): how steep, 0 to 1, and the way down it (unit, object space). */
export interface SandSlope {
  steep: Float;
  down: Vec3;
}

/**
 * Sand's relief finer than the mesh at unit direction p: wind ripples, combed by the wind (`combed`, 0 to 1: dug
 * sand has none until the wind combs it again), the lumps and crumbs of freshly `disturbed` sand, and streaks down
 * steep slopes where grains have slid. Returns a tone (about 1), how damp the crumbs are, and a bump height (orb
 * radii).
 */
export function sandRelief(
  p: Vec3,
  combed: Float,
  disturbed: Float = float(0),
  slope?: SandSlope,
): { tone: Float; damp: Float; height: Float } {
  const warp = octave(p, 14).mul(3).add(octave(p, 45).mul(0.3));
  const ripple = ripples(p, vec3(0.8, 0.1, -0.59), 95, warp);
  // a fresh dig throws out crumbs of damp sand from underneath, fewer the further from it
  const clumps = octave(p, 160).mul(0.5).add(0.5);
  const crumbs = smoothstep(
    float(1).sub(disturbed.mul(0.7)),
    float(1.05).sub(disturbed.mul(0.7)),
    clumps,
  ).mul(smoothstep(0.02, 0.15, disturbed));
  const lumps = octave(p, 210).mul(disturbed);
  let height = ripple.mul(0.0028).mul(combed).add(lumps.mul(0.0014)).add(crumbs.mul(0.0012));
  let tone = float(1).sub(ripple.mul(0.05).mul(combed));
  if (slope) {
    // grains that slid leave streaks down the slope: noise squeezed along the way down
    const q = p.mul(170);
    const streak = snoise(q.sub(slope.down.mul(dot(q, slope.down).mul(0.85))));
    const steep = smoothstep(0.35, 0.8, slope.steep);
    height = height.add(streak.mul(steep).mul(0.0012));
    tone = tone.add(streak.mul(steep).mul(0.05));
  }
  return { tone, damp: crumbs.mul(0.6), height: height.mul(BUMP / 0.0012) };
}

/** The colours a sand orb is made of: dry, a warmer and a paler drift, and damp. */
const DRY = color('#b89a6a');
const WARM = color('#c99d6a');
const PALE = color('#ddd0b0');
const WET = color('#7d6a4b');

/**
 * A whole orb of sand at unit direction p: faint drifts of warmer and paler sand under the grains; `wet` (0 to 1)
 * darkens and glosses it; `disturbed` (1 freshly moved, 0 settled) is the damper sand from underneath, darker,
 * lumpier, with the ripples wiped out; `below` (orb radii under the rest surface) damper still, the deeper it's
 * dug; `slope` where it stands steep.
 */
export function sandSurface(
  p: Vec3,
  wet: Float,
  disturbed: Float,
  light: SandLight,
  below: Float = float(0),
  slope?: SandSlope,
): { color: Vec3; roughness: Float; normal: Vec3; emissive: Vec3 } {
  const drift = octave(p, 3.5).add(octave(p, 9).mul(0.3));
  const relief = sandRelief(p, float(1).sub(disturbed), disturbed, slope);
  const base = mix(
    mix(DRY, WARM, smoothstep(0, 0.8, drift).mul(0.12)),
    PALE,
    smoothstep(0, 0.8, drift.negate()).mul(0.08),
  );
  // sand turned up from underneath is damp, and dries back as it settles
  const damp = max(max(wet, relief.damp), max(disturbed.mul(0.45), smoothstep(0, 0.02, below).mul(0.55)));
  const grains = sandGrains(p, bumped(relief.height.mul(float(1).sub(wet.mul(0.6)))), light, damp);
  return {
    color: mix(base, WET, damp).mul(relief.tone).mul(grains.tint),
    roughness: mix(mix(float(0.9), float(0.96), disturbed), float(0.45), wet),
    normal: grains.normal,
    emissive: grains.glint,
  };
}
