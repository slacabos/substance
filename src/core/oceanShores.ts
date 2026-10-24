import { Vector3 } from 'three';
import { dispersion, OCEAN_DEPTH, OCEAN_GRAVITY, OCEAN_WAVELENGTH_SCALE } from './oceanFFT';

export interface ShoreSite {
  readonly center: Vector3;
  readonly east: Vector3;
  readonly north: Vector3;
  readonly radius: number;
  readonly height: number;
  readonly seed: number;
}

const makeSite = (
  x: number,
  y: number,
  z: number,
  radius: number,
  height: number,
  seed: number,
): ShoreSite => {
  const center = new Vector3(x, y, z).normalize();
  const east = new Vector3(0, 1, 0).cross(center).normalize();
  const north = center.clone().cross(east).normalize();
  return { center, east, north, radius, height, seed };
};

/** Raised islands, with the largest coast visible from the initial camera angle. */
export const SHORE_SITES: readonly ShoreSite[] = [
  makeSite(0.48, 0.3, 0.82, 0.27, 0.17, 0.6),
  makeSite(-0.58, -0.12, 0.8, 0.2, 0.13, 1.8),
  makeSite(0.08, -0.68, 0.73, 0.17, 0.115, 3.4),
  makeSite(-0.24, 0.76, 0.6, 0.14, 0.1, 5.1),
  makeSite(0.3, -0.2, -0.93, 0.22, 0.14, 2.7),
];

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
  return (
    site.radius *
    (1 +
      0.17 * Math.cos(bearing * 2 + site.seed) +
      0.095 * Math.sin(bearing * 3 - site.seed * 1.7) +
      0.045 * Math.sin(bearing * 7 + site.seed * 2.3))
  );
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

/** A narrow wet bank rises into rock ledges and ridged high ground. */
export function shoreElevation(site: ShoreSite, dir: Vector3): number {
  const distance = shoreSiteDistance(site, dir);
  if (distance > 0) return 0.012 - 0.085 * smooth(distance / 0.045);
  const angle = Math.acos(Math.min(1, Math.max(-1, site.center.dot(dir))));
  const inland = smooth((site.radius * 1.04 - angle) / (site.radius * 0.85));
  const bank = smooth(-distance / 0.035);
  const u = site.east.dot(dir) / site.radius;
  const v = site.north.dot(dir) / site.radius;
  const ridge = 1 - Math.abs(shoreNoise(u * 3.2 + site.seed, v * 3.2, site.seed * 3));
  const detail = shoreNoise(u * 7, v * 7 + site.seed, site.seed * 2);
  const relief = 0.36 + ridge * 0.56 + detail * 0.18;
  const grain = shoreNoise(dir.x * 83, dir.y * 83, dir.z * 83);
  return 0.012 + 0.052 * bank + site.height * inland * bank * relief + grain * 0.008 * inland;
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
  return Math.min(OCEAN_DEPTH, Math.max(0.002, 0.085 * smooth(d / 0.045) - 0.012));
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

/** How far out from a shore the surf reaches before it's just the open sea's swell. */
export const SURF_REACH = 0.12;
