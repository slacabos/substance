import type { Texture } from 'three';
import type { Node } from 'three/webgpu';
import { LACE_RANGE } from '../core/foamLace';
import { SURF_FADE } from '../core/oceanShores';
import { OCEAN_DEFAULTS } from '../core/oceanTuning';
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
  mx_worley_noise_vec2,
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
const THREAD_STRETCH = 0.5;
/** How many foam cells across per thread: open patches a fifth of the surf zone across. */
const CELL_SCALE = 0.55;

/**
 * Surf foam as it is from the air: threads and lace rather than blobs. Ridged noise draws connected lines; squeezed
 * along the current `along` (unit, object space), so water running one way (the backwash) drags them into long
 * parallel streaks while churned water leaves them curled into lace. `amount` (0 to 1) is how much foam the sim
 * has here: a little is a few fine threads, more is a thicker, denser web; `fresh` (0 to 1) how freshly the water
 * was churned, the white front of a wave just broken.
 */
export function foamThreads(drift: FoamDrift, along: Vec3, amount: Float, fresh: Float): Float {
  // Read at both drifts and crossfaded (each drift restarts while it has no weight). Read once between them, the
  // pattern slid sideways back and forth as the blend went from one to the other.
  return mix(threadsAt(drift.b, along, amount, fresh), threadsAt(drift.a, along, amount, fresh), drift.blend);
}

/** The surf's foam pattern (foamThreads) at one drifted position `obj`. */
export function threadsAt(obj: Vec3, along: Vec3, amount: Float, fresh: Float): Float {
  const m = obj.mul(THREADS);
  // how the lace varies from place to place: fine and open here, coarse and dense there, streaked or curled
  const kind = snoise(m.mul(0.07).add(11.3));
  // squeezed along the current (more in some places than others), and warped hard by a slower noise, so the
  // lines are drawn out into marbled veins, as foam stirred by water is, rather than the even squiggles of one
  // noise
  const stretch = float(THREAD_STRETCH).sub(kind.mul(0.25));
  const squeezed = m.sub(along.mul(dot(m, along).mul(stretch)));
  const w = squeezed.mul(0.32);
  const warp = vec3(snoise(w), snoise(w.add(5.2)), snoise(w.add(9.1)));
  const bent = squeezed.add(warp.mul(float(2.6).add(kind.mul(1.2))));
  // Foam from the air is a net of cells: bubbles burst and drain, leaving the foam gathered along the borders of
  // the open patches between them, a web of filaments round open water. Drawn as the borders of Worley cells (the
  // gap between the nearest and second-nearest cell centres), warped by the water so they curl and swirl.
  const cells = mx_worley_noise_vec2(bent.mul(CELL_SCALE));
  const gap = cells.y.sub(cells.x);
  // a line of its own width, thicker where there's more foam, with soft edges a pixel wide
  const edge = fwidth(gap).add(0.01);
  // (and its width varies along it, where foam has gathered thicker)
  const gather = max(snoise(m.mul(0.9).add(4.4)), 0);
  const line = float(0.02).add(amount.mul(0.07)).add(kind.mul(0.02)).add(gather.mul(0.05));
  const lines = float(1).sub(smoothstep(line.sub(edge), line.add(edge), gap));
  // gathered into patches with open water between, more of it covered the fresher the foam
  const clumps = snoise(m.mul(0.11))
    .mul(0.6)
    .add(snoise(m.mul(0.27).add(3.3)).mul(0.4));
  // (and broken here and there, so it's torn lace rather than a maze of unbroken lines)
  const holes = snoise(m.mul(0.9));
  const threads = lines.mul(smoothstep(-0.7, -0.1, holes.add(kind.mul(0.3))));
  const held = smoothstep(0, 0.35, clumps.mul(0.5).add(amount.mul(2.6)).sub(0.05));
  // a wave's front, just broken, is thick white water torn by small holes, its edge torn at two scales (the finer
  // the lace's own) so it frays into threads and never shows the triangles the breaking is interpolated across
  // (torn hard: the breaking is interpolated linearly across the sim's triangles, and a cut through that alone
  // shows their edges as a sawtooth; the noise has to be what decides where the white ends)
  const torn = clumps.mul(0.3).add(holes.mul(0.22)).add(threads.mul(0.18));
  // (holes of two sizes, the finer one the lace's own cells, so it reads as churned white water, not a sheet)
  const pits = float(1)
    .sub(lines)
    .mul(smoothstep(0.1, 0.5, snoise(m.mul(2.3).add(1.7))));
  const front = smoothstep(0.25, 0.75, fresh.mul(0.8).add(torn).add(0.15)).mul(
    smoothstep(-0.35, 0.05, holes.add(clumps.mul(0.3))).mul(float(1).sub(pits.mul(0.85))),
  );
  // (fresh, thick foam fills in between the threads, but always with holes: lace, never a sheet)
  const fill = max(smoothstep(0.5, 0.9, amount.mul(held)).mul(smoothstep(-0.2, 0.3, holes)), front);
  // and old foam the backwash drags out to sea is drawn out into long fine streaks behind the white water
  const q = obj.mul(THREADS * 2.2);
  const streakAt = q.sub(along.mul(dot(q, along).mul(0.96))).add(warp.mul(1.2));
  const sr = float(1).sub(abs(snoise(streakAt)));
  const se = fwidth(sr).add(0.01);
  // (in patches, where a wave's foam was left behind, not over the whole surf zone)
  const trailing = smoothstep(0.02, 0.06, amount)
    .mul(float(1).sub(smoothstep(0.18, 0.4, amount)))
    .mul(smoothstep(0.05, 0.45, clumps));
  const streaks = smoothstep(float(0.93).sub(se), float(0.93).add(se), sr).mul(trailing).mul(0.6);
  return max(max(threads.mul(held), fill.mul(0.9)).mul(smoothstep(0.01, 0.1, amount)), streaks);
}

/**
 * The three foam layers, each with its own pattern, colour (linear) and default coverage (DeformUniforms'
 * uFoamWhitecap, uFoamAmbient and uFoamShore scale them):
 * - whitecaps: where the waves fold over and break, streaks along the crests; the brightest, freshest white;
 * - ambient: foam the water churned up (taps, drags, splashes, old whitecaps) and carries on its current, white
 *   water full of round bubbles thinning into a web; a little greyer, as thin foam lets the sea show through;
 * - shore: the surf zone's white water, a net of cells round open water with torn white fronts where waves
 *   break; a little warmer, carrying sand.
 */
export const FOAM_LAYERS = {
  whitecap: { coverage: 1, color: [0.95, 0.98, 0.98] },
  ambient: { coverage: 1, color: [0.86, 0.91, 0.92] },
  shore: { coverage: 1, color: [0.94, 0.94, 0.9] },
} as const;

const layerColor = (c: readonly number[]) => vec3(c[0]!, c[1]!, c[2]!);

/**
 * All the foam at a point: each layer's coverage (0 to 1), laid over one another (each covers what the others
 * leave open), and the colour they make together, each weighted by how much of it shows.
 */
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
): { amount: Float; color: Vec3 } {
  // the surf zone, where the shoreline layer takes over from the open sea's two
  const zone = shore && along ? float(1).sub(smoothstep(SURF_FADE * 0.6, SURF_FADE * 1.3, shore)) : float(0);
  const open = float(1).sub(zone);

  // whitecaps where the waves fold over, broken into streaks; the noise (the two drifted patterns crossfaded,
  // keeping its contrast) varies how readily each spot foams
  const wa = drift.blend;
  const wb = float(1).sub(wa);
  const n = foamNoise(drift.a, u)
    .sub(0.5)
    .mul(wa)
    .add(foamNoise(drift.b, u).sub(0.5).mul(wb))
    .div(sqrt(wa.mul(wa).add(wb.mul(wb))))
    .add(0.5);
  const whitecap = clamp(
    smoothstep(mix(0.3, 0.22, weather), 0.75, breaking.mul(n.mul(0.9).add(0.55)))
      .mul(u.uOceanFoam)
      .mul(u.uFoamWhitecap)
      .mul(open.mul(0.85).add(0.15)),
    0,
    1,
  );
  // ambient: churned foam the sim carries
  const ambient = clamp(
    foamLace(drift, lace, smoothstep(0.04, 0.6, simulated), u)
      .mul(u.uFoamAmbient)
      .mul(open),
    0,
    1,
  );
  // shoreline: the surf's white water
  const surf =
    shore && along
      ? clamp(
          foamThreads(drift, along, simulated, fresh)
            .mul(zone)
            .mul(u.uOceanFoam.div(OCEAN_DEFAULTS.foam))
            .mul(u.uFoamShore),
          0,
          1,
        )
      : float(0);

  // laid over one another
  const amount = float(1).sub(float(1).sub(whitecap).mul(float(1).sub(ambient)).mul(float(1).sub(surf)));
  const weight = max(whitecap.add(ambient).add(surf), 1e-4);
  const color = layerColor(FOAM_LAYERS.whitecap.color)
    .mul(whitecap)
    .add(layerColor(FOAM_LAYERS.ambient.color).mul(ambient))
    .add(layerColor(FOAM_LAYERS.shore.color).mul(surf))
    .div(weight);
  return { amount, color };
}

/** Approximate absorption through a deep body of water, plus a blue grazing-angle glint. */
export function oceanColor(
  wave: Float,
  simWave: Float,
  foam: Float,
  weather: Float,
  u: DeformUniforms,
  foamColor: Vec3 = layerColor(FOAM_LAYERS.whitecap.color),
): Vec3 {
  const facing = clamp(dot(normalViewGeometry, positionViewDirection), 0, 1);
  const depth = smoothstep(0.2, 0.9, facing);
  // a wave the hand raises lets more light through its crest and less through its trough
  const raised = smoothstep(-0.03, 0.03, simWave).sub(0.5).mul(0.45);
  const tint = clamp(smoothstep(-0.025, 0.025, wave).mul(0.35).add(depth.mul(0.1)).add(raised), 0, 1);
  const water = mix(u.uDeep, u.uShallow, tint).mul(float(1).sub(weather.mul(0.05)));
  const rim = pow(float(1).sub(facing), 5).mul(0.12);
  const lit = mix(water, vec3(0.08, 0.29, 0.39), rim);
  return mix(lit, foamColor, foam);
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
