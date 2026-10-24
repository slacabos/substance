import type { Node } from 'three/webgpu';
import {
  abs,
  attribute,
  clamp,
  cos,
  cross,
  dot,
  float,
  fract,
  fwidth,
  length,
  max,
  min,
  mix,
  normalize,
  normalView,
  positionGeometry,
  positionView,
  select,
  sign,
  sin,
  smoothstep,
  vec3,
} from 'three/tsl';
import { SHORE_WATERLINE, shelfLag } from '../core/oceanShores';
import { snoise } from './noise';
import { baseR } from './shape';
import { surfWave, type SurfInputs } from './surf';

type Float = Node<'float'>;
type Vec3 = Node<'vec3'>;

// Close-up detail for the coast's sand (materials/oceanShores.ts), which its vertices are far too coarse to carry:
// grain, wind ripples and a bump, darker and glossy where the per-vertex `terrain` attribute says the sand is wet,
// with the swash running over it.

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

/** Height the swash runs up a shore facing the swell, per unit of the sea's state, orb radii. */
const SWASH_RUNUP = 0.009;
/** How far up the beach the hand's waves run, per unit of the sea's height at the shoreline, and at most. */
const TAP_RUNUP = 1.5;
const MAX_RUNUP = 0.03;
/** Share of a wave's cycle the swash spends running up; the rest it drains back. */
const UPRUSH = 0.3;

/**
 * Swash: after each wave breaks at the shoreline (surfWave, as the sim's surf), a sheet of water runs up the beach
 * and drains back. Its reach is a height, so a gentle beach is covered far inland and a steep one barely wet.
 * Returns how much of a point the sheet covers, its depth there, the foam on it (a bubbly line at its front,
 * lace left behind as it drains), and how wet the run-up has left the sand.
 */
function swashAt(p: Vec3, height: Float, surf: SurfInputs, sea: Float) {
  const swash = attribute('swash', 'vec4');
  const foot = normalize(p.add(swash.xyz));
  const seaward = normalize(swash.xyz.add(p.cross(vec3(0, 0, 1)).mul(1e-5)));
  const wave = surfWave(foot, float(0), float(shelfLag(0)), swash.w, seaward, surf);
  // the cycle from the moment a crest reaches the shoreline
  const cycle = fract(wave.phase.sub(Math.PI / 2).div(Math.PI * 2));
  const reach = select(
    cycle.lessThan(UPRUSH),
    sin(cycle.div(UPRUSH).mul(Math.PI / 2)),
    cos(
      cycle
        .sub(UPRUSH)
        .div(1 - UPRUSH)
        .mul(Math.PI / 2),
    ).pow(1.5),
  );
  // each stretch of beach runs up its own distance: the front is ragged, not a contour line
  const ragged = snoise(foot.mul(45).add(swash.w))
    .mul(0.22)
    .add(snoise(foot.mul(130)).mul(0.12))
    .add(1);
  const runup = float(SWASH_RUNUP).mul(surf.energy).mul(wave.set).mul(wave.exposure).mul(ragged);
  // a wave the hand sent runs up the beach too, as high as the sea at the shoreline lifts it (it drains as
  // that sea falls back); the swell's swash otherwise
  // a wave the hand sent: the same ragged front, sheeting further up some stretches than others
  const handLevel = min(max(sea, 0).mul(TAP_RUNUP), MAX_RUNUP).mul(ragged);
  const level = max(runup.mul(reach), handLevel);
  const depth = max(level.sub(height), 0);
  const cover = float(1)
    .sub(smoothstep(level.sub(0.0004), level, height))
    .mul(smoothstep(0.0002, 0.0006, level));
  // bubbles drifting with the water: up the beach on the uprush, back down on the backwash
  const drift = p.add(seaward.mul(level.mul(-6)));
  const bubbles = smoothstep(
    -0.15,
    0.45,
    snoise(drift.mul(420))
      .mul(0.6)
      .add(snoise(drift.mul(1100)).mul(0.4)),
  );
  // the foam line is wider for a bigger wave, so a hand's surge brings a broad white front
  const band = max(level.mul(0.15), 0.0011);
  const front = float(1)
    .sub(smoothstep(0, band, abs(height.sub(level))))
    .mul(smoothstep(0.0002, 0.0008, level));
  // as it drains it leaves lace: patches of foam over the thin water behind the front
  const lace = smoothstep(
    0.05,
    0.45,
    snoise(drift.mul(230))
      .mul(0.7)
      .add(snoise(drift.mul(60)).mul(0.3)),
  );
  const draining = smoothstep(UPRUSH, UPRUSH + 0.25, cycle);
  const trail = cover
    .mul(float(1).sub(smoothstep(0, 0.004, depth)))
    .mul(draining)
    .mul(lace)
    .mul(0.8);
  const foam = clamp(front.mul(0.95).add(trail), 0, 1).mul(bubbles);
  // sand the run-up has reached stays dark and glossy for a while
  const soaked = float(1)
    .sub(smoothstep(runup.mul(0.7), runup.add(0.0006), height))
    .mul(0.8);
  return { cover, depth, foam, soaked };
}

/** The shores' surface: colour, roughness, and a bumped view-space normal, with the swash on them. */
export function shoreSurface(surf: SurfInputs, sea: Float): { color: Vec3; roughness: Float; normal: Vec3 } {
  const wet = attribute('terrain', 'float');
  const p = normalize(positionGeometry);

  // sand: fine grain, and faint ripples the wind combed across it
  const grain = octave(p, 900);
  const ripple = bands(p, vec3(0.8, 0.1, -0.59), 1400, octave(p, 60).mul(4));
  const tone = grain.mul(0.06).add(0.97).sub(ripple.mul(0.04));
  const height = grain
    .mul(0.25)
    .add(ripple.mul(0.35))
    .mul(float(1).sub(wet.mul(0.7)))
    .mul(BUMP);
  // height above the still sea, as the land mesh was built (it follows the cube morph like the water)
  const above = length(positionGeometry).sub(baseR(p)).sub(SHORE_WATERLINE);
  const sw = swashAt(p, above, surf, sea);
  const soaked = max(wet, sw.soaked);
  // wet sand is darker and glossy; under the swash, clear water over it, and white foam on top
  const ground = attribute('color', 'vec3')
    .mul(tone)
    .mul(mix(float(1), float(0.72), soaked));
  const water = mix(
    ground,
    ground.mul(vec3(0.6, 0.86, 0.88)).add(vec3(0.01, 0.05, 0.06)),
    // thin at the front, so mostly the wet sand shows; turquoise where it's deeper
    smoothstep(0, 0.003, sw.depth).mul(0.85),
  );
  const covered = mix(ground, water, sw.cover);
  return {
    color: mix(covered, vec3(0.92, 0.96, 0.97), sw.foam),
    roughness: mix(mix(float(0.88), float(0.35), soaked), float(0.08), sw.cover.mul(float(1).sub(sw.foam))),
    normal: bumped(height.mul(float(1).sub(sw.cover.mul(0.8)))),
  };
}
