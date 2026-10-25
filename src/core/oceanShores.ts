import { Vector3 } from 'three';
import { dispersion, OCEAN_DEPTH, OCEAN_GRAVITY, OCEAN_WAVELENGTH_SCALE } from './oceanFFT';

export interface ShoreSite {
  readonly center: Vector3;
  readonly east: Vector3;
  readonly north: Vector3;
  readonly radius: number;
  readonly height: number;
  readonly seed: number;
  /** How far the sand runs inland from the waterline before the dunes or rock begin, radians. */
  readonly beach: number;
}

const makeSite = (
  x: number,
  y: number,
  z: number,
  radius: number,
  height: number,
  seed: number,
  beach: number,
): ShoreSite => {
  const center = new Vector3(x, y, z).normalize();
  const east = new Vector3(0, 1, 0).cross(center).normalize();
  const north = center.clone().cross(east).normalize();
  return { center, east, north, radius, height, seed, beach };
};

/**
 * The land, all sand: one beach round the orb at a beach's scale, not a continent's. From the starting view the
 * dunes are at the top, the beach curves across the middle in a broad, shallow bay, and the sea is below, its
 * shallows running far out over a sandbar.
 */
export const SHORE_SITES: readonly ShoreSite[] = [makeSite(0, 0.988, 0.156, 1.49, 0.022, 0, 0.26)];

/** The land, whose side of the orb the swell runs toward. */
export const MAINLAND = SHORE_SITES[0]!;

/**
 * The seabed off the beach: it shelves gently out to the open sea's depth `width` radians out (as `power` of the
 * distance: 1 is a steady slope, which meets the beach's own gentle rise without a step at the waterline), with a
 * sandbar `bar.at` out, `bar.height` proud of the slope, which the waves break on before they reach the beach.
 */
export const SHELF = { width: 0.55, power: 1, bar: { at: 0.15, height: 0.008, width: 0.03 } } as const;
/** How far out the water still looks shallow: clear and turquoise over pale sand, well past the shelf's edge. */
export const SHALLOWS = 1.1;
/** The tallest a wave can stand over the still water's depth before it breaks (a breaker index). */
export const BREAKER = 0.5;

/** Height of the waterline's sand above the still sea, and how much a beach climbs to its top. */
export const SHORE_WATERLINE = 0.0015;
const WATERLINE = SHORE_WATERLINE;
const BEACH_RISE = 0.01;

const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));
const smooth = (x: number): number => {
  const t = clamp01(x);
  return t * t * (3 - 2 * t);
};

/** Coherent terrain grain without the regular bands produced by sine-wave textures. */
export function shoreNoise(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = smooth(x - ix);
  const fy = smooth(y - iy);
  const fz = smooth(z - iz);
  const hash = (a: number, b: number, c: number): number => {
    let h = Math.imul(a, 374761393) ^ Math.imul(b, 668265263) ^ Math.imul(c, 1442695041);
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
  };
  const mix = (a: number, b: number, t: number): number => a + (b - a) * t;
  const plane = (c: number): number =>
    mix(
      mix(hash(ix, iy, c), hash(ix + 1, iy, c), fx),
      mix(hash(ix, iy + 1, c), hash(ix + 1, iy + 1, c), fx),
      fy,
    );
  return mix(plane(iz), plane(iz + 1), fz) * 2 - 1;
}

/**
 * The shoreline's sweeps round a site, each `amplitude * cos(n * bearing + phase + seed * seedScale)` of its
 * radius: a beach's long, smooth sweeps, a broad bay facing the starting view (bearing -pi / 2), gentle points
 * either side, and no headlands.
 */
export const SHORE_SWEEPS = [
  { n: 2, amplitude: 0.12, phase: 0, seedScale: 1 },
  { n: 3, amplitude: 0.025, phase: 1.1 - Math.PI / 2, seedScale: -1.7 },
  { n: 5, amplitude: 0.012, phase: 0.4 - Math.PI / 2, seedScale: 2.3 },
] as const;

export function shoreRadius(site: ShoreSite, bearing: number): number {
  let r = 1;
  for (const s of SHORE_SWEEPS)
    r += s.amplitude * Math.cos(bearing * s.n + s.phase + site.seed * s.seedScale);
  return site.radius * r;
}

/** How wide the beach is at dir: wide sweeps of sand and narrow stretches along the same coast. */
export function beachWidth(site: ShoreSite, dir: Vector3): number {
  const bearing = Math.atan2(site.north.dot(dir), site.east.dot(dir));
  const along = shoreNoise(Math.cos(bearing) * 2.5 + site.seed, Math.sin(bearing) * 2.5, site.seed * 7);
  return site.beach * (1 + 0.6 * along);
}

/** Angular distance from the irregular shoreline: negative on land, positive in water. */
export function shoreSiteDistance(site: ShoreSite, dir: Vector3): number {
  const angle = Math.acos(Math.min(1, Math.max(-1, site.center.dot(dir))));
  const bearing = Math.atan2(site.north.dot(dir), site.east.dot(dir));
  return angle - shoreRadius(site, bearing);
}

export function shoreDistance(dir: Vector3): number {
  let nearest = Infinity;
  for (const site of SHORE_SITES) nearest = Math.min(nearest, shoreSiteDistance(site, dir));
  return nearest;
}

/**
 * How far the seabed lies below the waterline's sand at distance d out (d >= 0): the shelf's slope less the bar,
 * then on past the shelf down to the deep sea floor.
 */
export function shelfDrop(d: number): number {
  const { width, power, bar } = SHELF;
  const slope = (OCEAN_DEPTH + WATERLINE) * Math.min(1, d / width) ** power;
  const ridge = bar.height * Math.exp(-(((d - bar.at) / bar.width) ** 2));
  return slope - ridge + 0.035 * smooth((d - width) / 0.15);
}

/**
 * The ground at dir: offshore the seabed shelves away over the bar; ashore a beach climbs gently from the
 * waterline to a line of dunes, with lower dunes and rolling sand behind them.
 */
export function shoreElevation(site: ShoreSite, dir: Vector3): number {
  const distance = shoreSiteDistance(site, dir);
  if (distance > 0) return WATERLINE - shelfDrop(distance);
  const beach = beachWidth(site, dir);
  const sand = WATERLINE + BEACH_RISE * smooth(-distance / beach);
  const angle = Math.acos(Math.min(1, Math.max(-1, site.center.dot(dir))));
  const inland = smooth((site.radius * 1.04 - angle) / (site.radius * 0.85));
  // past the beach the land rises: quickly on a small island, over a long slope inland on the continent
  const bank = smooth((-distance - beach * 0.8) / Math.max(0.035, site.radius * 0.1));
  // all sand: broad rolling hills, crossed by dune ridges with soft crests
  const hills = 0.55 + 0.45 * shoreNoise(dir.x * 4 + site.seed, dir.y * 4, dir.z * 4);
  const ridge = 1 - Math.abs(shoreNoise(dir.x * 26 + site.seed, dir.y * 26, dir.z * 26 - site.seed));
  const dunes = ridge * ridge * (0.006 + 0.008 * inland);
  // the foredune: a line of dunes along the top of the beach, higher in some stretches than others
  const along = 0.65 + 0.35 * shoreNoise(dir.x * 9 + 2.7, dir.y * 9, dir.z * 9);
  const foredune = 0.011 * along * Math.exp(-(((-distance - beach * 1.15) / 0.035) ** 2));
  return sand + foredune + bank * (dunes + site.height * (0.25 + 0.75 * inland * hills));
}

/** Ground height used by spray collision; distant seabed sits well below the surface. */
export function shoreGroundHeight(dir: Vector3): number {
  let height = -Infinity;
  for (const site of SHORE_SITES) height = Math.max(height, shoreElevation(site, dir));
  return height;
}

/**
 * Still water over the seabed off a shore, at angular distance d (shelfDrop): a few millimetres over the bar
 * and inshore, deepening slowly to the open sea's depth at the shelf's edge.
 */
export function shelfDepth(d: number): number {
  return Math.min(OCEAN_DEPTH, Math.max(0.002, shelfDrop(Math.max(0, d)) - WATERLINE));
}

/** Where the shelf ends and the open sea's depth begins. */
export const SHELF_EDGE = (() => {
  let d = 0;
  while (shelfDepth(d) < OCEAN_DEPTH) d += 1e-4;
  return d;
})();

/**
 * How long a crest takes to cross the shelf from distance d out to its edge, per unit of frequency: the
 * integral of 1 / sqrt(g H) (shallow-water speed). Times the swell's frequency, it's the phase a crest gains
 * by slowing and bunching up as the water shallows.
 */
export function shelfLag(d: number): number {
  if (d >= SHELF_EDGE) return 0;
  const from = Math.max(0, d);
  const steps = 64;
  const step = (SHELF_EDGE - from) / steps;
  let sum = 0;
  for (let i = 0; i < steps; i++)
    sum += step / Math.sqrt(OCEAN_GRAVITY * shelfDepth(from + (i + 0.5) * step));
  return sum;
}

/** The swell that breaks on the shores: the FFT sea's peak, its frequency and its wavenumber in open water. */
export function shoreSwell(peakWavelength: number): { omega: number; k: number } {
  const k = (2 * Math.PI) / (peakWavelength * OCEAN_WAVELENGTH_SCALE);
  return { omega: dispersion(k), k };
}

export interface ShoreSurfSample {
  height: number;
  breaking: number;
}

/**
 * The surf is several trains of swell, not one sine: each of its own period (`ratio` of the main swell's frequency;
 * deep-water waves of that frequency are `ratio`² as short), its own slight slant to the beach (`slant` crests a
 * turn round the land, a whole number so the phase joins up all the way round), and its own share of the height.
 * Where their crests fall together a big wave breaks, where they don't a small one: the waves come in all sizes,
 * and no two along the beach alike.
 */
export const SURF_TRAINS = [
  { ratio: 1, slant: 2, share: 1, seed: 0 },
  { ratio: 0.83, slant: -3, share: 0.6, seed: 1.9 },
  { ratio: 1.21, slant: 5, share: 0.45, seed: 4.3 },
] as const;
export type SurfTrain = (typeof SURF_TRAINS)[number];
export const SURF_SLANT = SURF_TRAINS[0].slant;
const SHARES = SURF_TRAINS.reduce((sum, t) => sum + t.share, 0);

/**
 * How far a train's crests are bent along the beach at `bearing`, radians of phase: the seabed's own unevenness,
 * which brings them in early here and late there, so a crest is a ragged line rather than a ruler.
 */
export function surfBend(train: SurfTrain, bearing: number): number {
  return 0.9 * Math.sin(3 * bearing + train.seed) + 0.5 * Math.sin(7 * bearing + 2.3 * train.seed);
}

/**
 * A train's phase at a point off the beach, `distance` out from it and `bearing` round the land's centre. Swell
 * refracts until its crests run nearly parallel to the shore, so they come straight in at the beach: offshore at
 * the train's own k per radian of distance, and on the shelf slowing and crowding together as the water shallows
 * (`lag`, the shelf's delay, times its omega), slanted and bent a little along the beach. `clock` is the train's
 * omega times time, added up frame by frame.
 */
export function shoreSurfPhase(
  bearing: number,
  distance: number,
  lag: number,
  swell: { omega: number; k: number },
  clock: number,
  train: SurfTrain = SURF_TRAINS[0],
): number {
  const omega = swell.omega * train.ratio;
  const k = swell.k * train.ratio * train.ratio;
  return (
    omega * lag -
    k * Math.max(0, distance - SHELF_EDGE) +
    train.slant * bearing +
    surfBend(train, bearing) -
    clock
  );
}

/**
 * How lopsided a wave is at distance d out, 0 to about 0.8: inside the bar it's a breaking bore, a steep foaming
 * front with a long gentle back, rather than a sine.
 */
export function surfSkew(d: number): number {
  return 0.8 * (1 - smooth((d - SHELF.bar.at - 0.02) / 0.08));
}

/** A wave's shape at `phase`: a sine, sheared by `skew` into a steep front and a long back. */
export function surfShape(phase: number, skew: number): number {
  return Math.sin(phase + skew * Math.sin(phase));
}

/**
 * Where the surf breaks at distance d out, 0 to 1: hardest on the bar and again at the beach, and the broken
 * waves running on between them as whitewater.
 */
export function surfBreakZone(d: number): number {
  const { bar } = SHELF;
  const beach = 1 - smooth((d - 0.012) / 0.06);
  const onBar = 0.9 * Math.exp(-(((d - bar.at) / bar.width) ** 2));
  const inside = 0.45 * (1 - smooth((d - bar.at) / 0.04));
  return Math.max(beach, onBar, inside);
}

/**
 * Surf at a point with the given phase: the swell rises as it shoals (Green's law, height going as depth to
 * the -1/4), steepens into a sharp crest, and breaks on the bar and the beach. Shores facing the swell take it
 * all; lee shores only what bends round to them. `energy` is the sea's state (1 at the tuned wind and amplitude).
 */
export function sampleShoreSurf(
  phase: number,
  distance: number,
  facing: number,
  energy: number,
  out: ShoreSurfSample,
): void {
  if (distance < 0 || distance > SURF_REACH) {
    out.height = 0;
    out.breaking = 0;
    return;
  }
  const envelope =
    (1 - smooth((distance - SURF_FADE) / (SURF_REACH - SURF_FADE))) *
    Math.min(1.8, (OCEAN_DEPTH / shelfDepth(distance)) ** 0.25) *
    (0.65 + smooth(distance / 0.025) * 0.35);
  const exposure = 0.15 + 0.85 * smooth((facing + 0.2) / 0.8);
  const s = surfShape(phase, surfSkew(distance));
  const crest = Math.max(0, s) ** 5;
  // no taller than the water's depth lets it stand, so its trough never bares the seabed
  const amplitude = Math.min(SURF_HEIGHT * envelope * exposure * energy, BREAKER * shelfDepth(distance));
  // (it flattens out toward the shoreline, troughs most: there the water would lie nearly along the beach's own
  // slope, and its edge would wander far up and down it with every small difference between the sim's vertices,
  // a sharp crest most of all; the swash, drawn on the sand, is the run-up and the backwash)
  const height = (s * 0.32 + crest * 0.95 - 0.17) * amplitude;
  const calm = smooth((distance - 0.005) / 0.035);
  out.height = height * (height < 0 ? calm : 0.3 + 0.7 * calm);
  out.breaking = crest * surfBreakZone(distance) * exposure * energy;
}

/**
 * The whole surf at a point: every train (SURF_TRAINS) sampled at its own phase and clock (`clocks`, one per train)
 * and added up by its share. They're averaged, so the water can never stand taller than one train lets it; where
 * their crests fall together a big wave breaks.
 */
export function shoreSurfAt(
  bearing: number,
  distance: number,
  lag: number,
  swell: { omega: number; k: number },
  clocks: readonly number[],
  energy: number,
  out: ShoreSurfSample,
): void {
  const one = { height: 0, breaking: 0 };
  let height = 0;
  let breaking = 0;
  SURF_TRAINS.forEach((train, t) => {
    const phase = shoreSurfPhase(bearing, distance, lag, swell, clocks[t]!, train);
    sampleShoreSurf(phase, distance, 1, energy, one);
    height += one.height * train.share;
    breaking = Math.max(breaking, one.breaking * Math.min(1, train.share * 1.4));
  });
  // averaged, then lifted a little: trains rarely all fall together
  out.height = Math.max((height / SHARES) * 1.35, -BREAKER * shelfDepth(Math.max(0, distance)));
  out.breaking = breaking;
}

/**
 * Waves per set. Swell is many trains of slightly different periods: where they fall into step the waves add up
 * into a set of bigger ones, where they don't there's a lull. At the swell's 7 s period that's about a minute.
 */
export const SURF_SET_WAVES = 9;

/**
 * How big the surf is at this point of a set: about 0.55 in a lull to 1.45 at a set's heart, 0.9 on average.
 * `spatial` is the surf's phase without its clock, so a set travels in with the crests; `setClock` runs at the
 * swell's frequency over SURF_SET_WAVES, and `seed` puts each island's sets at their own times.
 */
export function surfSet(spatial: number, setClock: number, seed: number): number {
  const s = 0.5 + 0.5 * Math.sin(spatial / SURF_SET_WAVES - setClock + seed);
  return 0.55 + 0.9 * s * s;
}

/** How far out from a shore the surf reaches before it's just the open sea's swell, and where it starts to fade. */
export const SURF_REACH = 0.4;
export const SURF_FADE = SHELF.bar.at + 0.05;
/** The surf's height scale, orb radii: big next to the orb, as breakers are next to a beach. */
export const SURF_HEIGHT = 0.034;
