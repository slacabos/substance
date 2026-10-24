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
 * The land, all sand: a continent covering about half the orb, its coast in view from the start, with deep bays
 * and headlands, wide beaches and dunes; and two small sandy islands out in the open half.
 */
export const SHORE_SITES: readonly ShoreSite[] = [
  makeSite(0.9, 0.35, -0.25, 1.42, 0.04, 0.6, 0.16),
  makeSite(-0.35, -0.3, 0.88, 0.1, 0.02, 3.4, 0.05),
  makeSite(-0.7, 0.55, 0.3, 0.07, 0.016, 5.1, 0.05),
];

/** The continent, whose side of the orb the swell runs toward. */
export const MAINLAND = SHORE_SITES[0]!;

/** Height of the waterline's sand above the still sea, and how much a beach climbs to its top. */
export const SHORE_WATERLINE = 0.004;
const WATERLINE = SHORE_WATERLINE;
const BEACH_RISE = 0.012;

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

export function shoreRadius(site: ShoreSite, bearing: number): number {
  // broad lobes in proportion to the site, and finer bays and headlands of their own size (but no bigger than a
  // small island can take)
  const fine = Math.min(site.radius, 0.3) ** 2 / 0.3;
  return (
    site.radius *
      (1 +
        0.17 * Math.cos(bearing * 2 + site.seed) +
        0.095 * Math.sin(bearing * 3 - site.seed * 1.7) +
        0.045 * Math.sin(bearing * 7 + site.seed * 2.3)) +
    fine * (0.1 * Math.sin(bearing * 13 + site.seed * 3.1) + 0.05 * Math.sin(bearing * 29 - site.seed))
  );
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
 * The ground at dir: offshore the seabed shelves away; ashore a beach climbs gently from the waterline, then
 * dunes or rock rise toward the island's ridged heart, as high as the site allows.
 */
export function shoreElevation(site: ShoreSite, dir: Vector3): number {
  const distance = shoreSiteDistance(site, dir);
  if (distance > 0) return WATERLINE - 0.085 * smooth(distance / 0.045);
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
  return sand + bank * (dunes + site.height * (0.25 + 0.75 * inland * hills));
}

/** Ground height used by spray collision; distant seabed sits well below the surface. */
export function shoreGroundHeight(dir: Vector3): number {
  let height = -0.085;
  for (const site of SHORE_SITES) {
    if (shoreSiteDistance(site, dir) < 0.05) height = Math.max(height, shoreElevation(site, dir));
  }
  return height;
}

/**
 * Still water over the seabed off a shore, at angular distance d (the bank's profile in shoreElevation): a
 * narrow shelf that reaches the open sea's depth a few hundredths of a radian out.
 */
export function shelfDepth(d: number): number {
  return Math.min(OCEAN_DEPTH, Math.max(0.002, 0.085 * smooth(d / 0.045) - WATERLINE));
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
 * The surf's phase at a point off a shore. Offshore it's the swell's own, a plane wave of wavenumber k along
 * the swell's direction (`along` is the point's distance in that direction from its island's centre). On the
 * shelf the crests slow and crowd together as the water shallows; the along-shore part of the wavenumber is
 * kept (Snell's law), so crests swing round to meet the shore and break along it in turn rather than all at once.
 * `facing` is the cosine between the swell's direction and the way into the shore; `clock` is omega times time.
 */
export function shoreSurfPhase(
  along: number,
  distance: number,
  lag: number,
  facing: number,
  swell: { omega: number; k: number },
  clock: number,
): number {
  const across = Math.max(0, SHELF_EDGE - distance);
  return swell.k * along + swell.omega * lag - swell.k * Math.max(facing, 0.15) * across - clock;
}

/**
 * Surf at a point with the given phase: the swell rises as it shoals (Green's law, height going as depth to
 * the -1/4), steepens into a sharp crest, and breaks on the bank. Shores facing the swell take it all; lee
 * shores only what bends round to them. `energy` is the sea's state (1 at the tuned wind and amplitude).
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
    (1 - smooth((distance - 0.04) / (SURF_REACH - 0.04))) *
    Math.min(1.8, (OCEAN_DEPTH / shelfDepth(distance)) ** 0.25) *
    (0.65 + smooth(distance / 0.025) * 0.35);
  const exposure = 0.15 + 0.85 * smooth((facing + 0.2) / 0.8);
  const s = Math.sin(phase);
  const crest = Math.max(0, s) ** 5;
  out.height = (s * 0.32 + crest * 0.95 - 0.17) * 0.03 * envelope * exposure * energy;
  out.breaking = crest * (1 - smooth((distance - 0.012) / 0.06)) * exposure * energy;
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

/** How far out from a shore the surf reaches before it's just the open sea's swell. */
export const SURF_REACH = 0.12;
