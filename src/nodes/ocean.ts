import type { Texture } from 'three';
import type { Node } from 'three/webgpu';
import { LACE_RANGE } from '../core/foamLace';
import { SURF_FADE } from '../core/oceanShores';
import {
  cameraViewMatrix,
  abs,
  clamp,
  dot,
  float,
  floor,
  fract,
  fwidth,
  length,
  max,
  mix,
  normalize,
  normalViewGeometry,
  positionViewDirection,
  pow,
  smoothstep,
  sqrt,
  texture,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { DeformUniforms } from './deform';
import { snoise } from './noise';

type Vec3 = Node<'vec3'>;
type Float = Node<'float'>;

/** Wavelength the foam pattern's scale was authored at. */
const FOAM_REFERENCE_WAVELENGTH = 0.33;

function h31(x: Vec3): Float {
  const p = fract(x.mul(0.3183099).add(0.1)).mul(17);
  return fract(p.x.mul(p.y).mul(p.z).mul(p.x.add(p.y).add(p.z)));
}

function vnoise(x: Vec3): Float {
  const i = floor(x);
  const f0 = fract(x);
  const f = f0.mul(f0).mul(f0.mul(-2).add(3));
  const c = (dx: number, dy: number, dz: number) => h31(i.add(vec3(dx, dy, dz)));
  return mix(
    mix(mix(c(0, 0, 0), c(1, 0, 0), f.x), mix(c(0, 1, 0), c(1, 1, 0), f.x), f.y),
    mix(mix(c(0, 0, 1), c(1, 0, 1), f.x), mix(c(0, 1, 1), c(1, 1, 1), f.x), f.y),
    f.z,
  );
}

/** Where the foam pattern is read: two drifted positions and how much of it follows the first (see OceanSim). */
export interface FoamDrift {
  a: Vec3;
  b: Vec3;
  blend: Float;
}

/** Foam lace cells per orb radius at the authored wavelength: bubbles a few pixels across. */
const LACE = 40;
/** Cells across the baked lace tile (core/foamLace.ts). */
export const LACE_TILE_CELLS = 32;

/**
 * The baked bubbles at obj (core/foamLace.ts): x the distance to the nearest bubble's centre, in cell widths; y
 * that bubble's own random number. Projected along the three axes and blended by how squarely each faces the
 * surface; each bubble size is offset so they never line up.
 */
function bubblesAt(lace: Texture, obj: Vec3, frequency: number, scale: Float): Node<'vec2'> {
  const n = normalize(obj);
  const w = pow(abs(n), vec3(6));
  const weights = w.div(w.x.add(w.y).add(w.z));
  const at = obj.mul(scale.mul(frequency / LACE_TILE_CELLS)).add(frequency * 0.37);
  const read = (uv: Node<'vec2'>) => texture(lace, uv).xy;
  const sample = read(at.yz).mul(weights.x).add(read(at.zx).mul(weights.y)).add(read(at.xy).mul(weights.z));
  return vec2(sample.x.mul(LACE_RANGE), sample.y);
}

/**
 * Churned foam: white water full of round holes. Where the sim has foam, a slow noise gathers it into clumps
 * with open water between. Within them each bubble opens to its own radius: pinholes in thick foam, growing and
 * merging into a thin, translucent web as it thins, and a few large bubbles open wide. Each bubble size is read
 * at both drifts and crossfaded before the holes are cut, so bubbles never double up. Already scaled by the
 * foam strength.
 */
function foamLace(drift: FoamDrift, lace: Texture, amount: Float, u: DeformUniforms): Float {
  const scale = float(FOAM_REFERENCE_WAVELENGTH).div(u.uOceanWavelength).mul(LACE);
  const middle = mix(drift.b, drift.a, drift.blend).mul(scale);
  const clumps = vnoise(middle.mul(0.08))
    .mul(0.65)
    .add(vnoise(middle.mul(0.21).add(7.1)).mul(0.35));
  const held = clamp(amount.mul(clumps.sub(0.3).mul(2.6)), 0, 1);
  const thin = float(1).sub(held);
  const bubbles = (frequency: number) =>
    mix(bubblesAt(lace, drift.b, frequency, scale), bubblesAt(lace, drift.a, frequency, scale), drift.blend);
  // 1 outside a round hole of the given radius around each bubble's centre
  const outside = (distance: Float, radius: Float) => {
    const width = fwidth(distance).add(0.015);
    return smoothstep(radius.sub(width), radius.add(width), distance);
  };
  const holes = (b: Node<'vec2'>, smallest: number, largest: number) =>
    outside(b.x, mix(smallest, largest, b.y).mul(thin.mul(0.9).add(0.45)));
  const small = bubbles(2.7);
  const foam = holes(bubbles(1), 0.12, 0.5).mul(holes(small, 0.08, 0.42));
  // a few large bubbles open wide, more as the foam thins; their rims are ragged with the small bubbles
  const large = bubbles(0.38);
  const opening = smoothstep(0.6, 0.95, large.y).mul(thin.mul(0.6).add(0.4)).mul(0.7);
  const open = float(1)
    .sub(outside(large.x.add(small.x.mul(0.12)), opening))
    .mul(smoothstep(0, 0.05, opening));
  const coverage = foam.mul(float(1).sub(open)).mul(smoothstep(0.01, 0.12, held));
  // Applied after the foam strength, so it can't push churned foam back to solid white: thin foam is a
  // translucent film, and even the thickest lets a little water through, more under some bubbles than others.
  const film = mix(0.4, 0.85, held).mul(mix(0.85, 1, small.y));
  return clamp(coverage.mul(u.uOceanFoam), 0, 1).mul(film);
}

/** Two octaves of value noise, each fading to its mean once its cells get smaller than a pixel. */
function foamNoise(obj: Vec3, u: DeformUniforms): Float {
  // Narrow in the swell direction and stretched along each crest.
  const scale = float(FOAM_REFERENCE_WAVELENGTH).div(u.uOceanWavelength);
  const q = vec3(
    dot(obj, vec3(0.35, 0.9, 0.25))
      .mul(79)
      .mul(scale)
      .sub(u.uFoamScroll.mul(1.3)),
    dot(obj, vec3(-0.88, 0.32, 0.34))
      .mul(17)
      .mul(scale)
      .add(u.uFoamScroll.mul(0.18)),
    dot(obj, vec3(0.3, -0.16, 0.94))
      .mul(12)
      .mul(scale),
  );
  const cells = length(fwidth(q));
  const octave = (frequency: number) =>
    mix(vnoise(q.mul(frequency)), float(0.5), smoothstep(0.35, 0.8, cells.mul(frequency)));
  return octave(1).mul(0.7).add(octave(2.35).mul(0.3));
}

/** Threads per orb radius across the current, and how much longer than wide they're drawn along it. */
const THREADS = 11;
const THREAD_STRETCH = 0.88;

/**
 * Surf foam as it is from the air: threads and lace rather than blobs. Ridged noise draws connected lines; squeezed
 * along the current `along` (unit, object space), so water running one way (the backwash) drags them into long
 * parallel streaks while churned water leaves them curled into lace. `amount` (0 to 1) is how much foam the sim
 * has here: a little is a few fine threads, more is a thicker, denser web; `fresh` (0 to 1) how freshly the water
 * was churned, the white front of a wave just broken. Read between the two drifts, so the pattern rides the water.
 */
export function foamThreads(drift: FoamDrift, along: Vec3, amount: Float, fresh: Float): Float {
  // (read once, between the drifts, which glides without jumps: the restarting drift has no weight)
  const middle = mix(drift.b, drift.a, drift.blend);
  const m = middle.mul(THREADS);
  // how the lace varies from place to place: fine and open here, coarse and dense there, streaked or curled
  const kind = snoise(m.mul(0.07).add(11.3));
  const at = (obj: Vec3) => {
    const q = obj.mul(THREADS);
    // squeezed along the current (more in some places than others), and bent so the streaks curl and fork
    const stretch = float(THREAD_STRETCH).sub(kind.mul(0.25));
    const squeezed = q.sub(along.mul(dot(q, along).mul(stretch)));
    const bend = snoise(q.mul(0.13));
    const bent = squeezed.add(vec3(bend, bend.negate(), snoise(q.mul(0.13).add(5.2))).mul(1.6));
    const ridge = (k: number) => {
      const p = bent.mul(k).add(k * 3.1);
      const r = float(1).sub(abs(snoise(p)));
      // a line of its own width, thicker where there's more foam, with soft edges a pixel wide
      const edge = fwidth(r).add(0.01);
      const line = float(0.07).add(amount.mul(0.16)).add(kind.mul(0.08));
      return smoothstep(float(1).sub(line).sub(edge), float(1).sub(line).add(edge), r);
    };
    return max(ridge(1), ridge(2.1).mul(smoothstep(-0.6, 0.4, kind).mul(0.9)));
  };
  // (and broken here and there, so it's torn lace rather than a maze of unbroken lines)
  const threads = at(middle).mul(smoothstep(-0.55, 0.1, snoise(middle.mul(THREADS * 1.9).add(2.2))));
  // gathered into patches with open water between, more of it covered the fresher the foam; thick foam (fresh
  // white water) thickens the threads into a dense web with holes, never a flat sheet
  const clumps = snoise(m.mul(0.11))
    .mul(0.6)
    .add(snoise(m.mul(0.27).add(3.3)).mul(0.4));
  // (however thick, a third of it stays open: white water is always torn)
  const held = smoothstep(0, 0.3, clumps.add(amount.mul(0.9)).sub(0.55));
  // (fresh, thick foam fills in between the threads, but always with holes: lace, never a sheet)
  // a wave's front, just broken, is thick white water torn by small holes, and its ragged edge trails threads
  const holes = snoise(m.mul(0.9));
  // (its edge torn by noise, so it never shows the triangles the breaking is interpolated across)
  const torn = snoise(m.mul(0.45).add(7.7)).mul(0.2).add(holes.mul(0.08));
  const front = smoothstep(0.12, 0.42, fresh.add(torn)).mul(
    smoothstep(-0.6, -0.2, holes.add(clumps.mul(0.3))),
  );
  const fill = max(smoothstep(0.5, 0.9, amount.mul(held)).mul(smoothstep(-0.2, 0.3, holes)), front);
  // and old foam the backwash drags out to sea is drawn out into long fine streaks behind the white water
  const streak = (obj: Vec3) => {
    const q = obj.mul(THREADS * 2.2);
    const squeezed = q.sub(along.mul(dot(q, along).mul(0.96))).add(snoise(q.mul(0.05)).mul(1.2));
    const r = float(1).sub(abs(snoise(squeezed)));
    const edge = fwidth(r).add(0.01);
    return smoothstep(float(0.93).sub(edge), float(0.93).add(edge), r);
  };
  const trailing = smoothstep(0.02, 0.08, amount)
    .mul(float(1).sub(smoothstep(0.25, 0.55, amount)))
    .mul(smoothstep(-0.3, 0.3, clumps.add(0.2)));
  const streaks = streak(middle).mul(trailing).mul(0.8);
  return max(max(threads.mul(held), fill.mul(0.9)).mul(smoothstep(0.03, 0.2, amount)), streaks);
}

/** Whitecaps follow steep travelling crests; simulated foam persists and drifts after impacts. */
export function oceanFoam(
  drift: FoamDrift,
  lace: Texture,
  breaking: Float,
  weather: Float,
  simulated: Float,
  u: DeformUniforms,
  shore?: Float,
  along?: Vec3,
  fresh: Float = float(0),
): Float {
  // the two drifted patterns crossfaded, keeping the noise's contrast through the blend
  const wa = drift.blend;
  const wb = float(1).sub(wa);
  const n = foamNoise(drift.a, u)
    .sub(0.5)
    .mul(wa)
    .add(foamNoise(drift.b, u).sub(0.5).mul(wb))
    .div(sqrt(wa.mul(wa).add(wb.mul(wb))))
    .add(0.5);
  // Whitecaps where the waves fold over, broken into streaks; the noise varies how readily each spot foams.
  const crests = smoothstep(mix(0.3, 0.22, weather), 0.75, breaking.mul(n.mul(0.9).add(0.55)));
  // churned foam left by the sim: taps, drags, splashes and old whitecaps
  let sprayFoam = foamLace(drift, lace, smoothstep(0.04, 0.6, simulated), u);
  let foam = crests;
  if (shore && along) {
    // the surf zone's white water: threads and streaks the waves leave and the water drags, in place of the open
    // sea's round-bubbled lace
    const zone = float(1).sub(smoothstep(SURF_FADE * 0.6, SURF_FADE * 1.3, shore));
    foam = max(foam.mul(float(1).sub(zone.mul(0.85))), foamThreads(drift, along, simulated, fresh).mul(zone));
    sprayFoam = sprayFoam.mul(float(1).sub(zone));
  }
  return max(clamp(foam.mul(u.uOceanFoam), 0, 1), sprayFoam);
}

/** Approximate absorption through a deep body of water, plus a blue grazing-angle glint. */
export function oceanColor(
  wave: Float,
  simWave: Float,
  foam: Float,
  weather: Float,
  u: DeformUniforms,
): Vec3 {
  const facing = clamp(dot(normalViewGeometry, positionViewDirection), 0, 1);
  const depth = smoothstep(0.2, 0.9, facing);
  // a wave the hand raises lets more light through its crest and less through its trough
  const raised = smoothstep(-0.03, 0.03, simWave).sub(0.5).mul(0.45);
  const tint = clamp(smoothstep(-0.025, 0.025, wave).mul(0.35).add(depth.mul(0.1)).add(raised), 0, 1);
  const water = mix(u.uDeep, u.uShallow, tint).mul(float(1).sub(weather.mul(0.05)));
  const rim = pow(float(1).sub(facing), 5).mul(0.12);
  const lit = mix(water, vec3(0.08, 0.29, 0.39), rim);
  return mix(lit, vec3(0.93, 0.97, 0.965), foam);
}

/**
 * The sun on the water, added as emission. A GGX glint whose width follows the surface roughness, so waves too
 * small to see spread it into a glitter path; light scattered back out of thin crests seen against the sun
 * (after Atlas, GDC 2019); and sunlit foam. `normal` is the view-space shading normal, `height` the wave height.
 */
export function oceanSunLight(
  normal: Vec3,
  height: Float,
  foam: Float,
  roughness: Float,
  u: DeformUniforms,
): Vec3 {
  const n = normalize(normal);
  const v = positionViewDirection;
  const l = normalize(cameraViewMatrix.mul(vec4(u.uSunDir, 0)).xyz);
  const h = normalize(l.add(v));
  const nl = clamp(dot(n, l), 0, 1);
  const nv = clamp(dot(n, v), 1e-3, 1);
  const nh = clamp(dot(n, h), 0, 1);
  // the sun's disc keeps even glassy water's glint a little wide
  const alpha = roughness.mul(roughness);
  const a2 = max(alpha.mul(alpha), 0.0004);
  const lobe = nh.mul(nh).mul(a2.sub(1)).add(1);
  const distribution = a2.div(lobe.mul(lobe).mul(Math.PI));
  const visibility = float(0.5).div(
    nl
      .mul(sqrt(nv.mul(nv).mul(float(1).sub(a2)).add(a2)))
      .add(nv.mul(sqrt(nl.mul(nl).mul(float(1).sub(a2)).add(a2))))
      .add(1e-5),
  );
  const fresnel = pow(float(1).sub(clamp(dot(v, h), 0, 1)), 5)
    .mul(0.98)
    .add(0.02);
  const glint = distribution.mul(visibility).mul(fresnel).mul(nl);

  const crest = smoothstep(0.002, 0.012, height);
  const towardSun = pow(clamp(dot(v, l.negate()), 0, 1), 4);
  const litFromBehind = pow(clamp(dot(l, n).mul(-0.5).add(0.5), 0, 1), 3);
  const scatter = crest
    .mul(towardSun)
    .mul(litFromBehind)
    .mul(0.8)
    .add(crest.mul(nv.mul(nv)).mul(0.03))
    .add(nl.mul(0.015));
  const water = vec3(0.07, 0.36, 0.33).mul(scatter).add(vec3(glint));
  return mix(water, vec3(nl.mul(0.25)), foam).mul(u.uSunColor);
}
