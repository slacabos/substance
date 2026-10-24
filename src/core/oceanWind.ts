import { Vector3 } from 'three';
import { OCEAN_WIND_ANGLE } from './oceanFFT';
import { OCEAN_DEFAULTS } from './oceanTuning';

// Real wind is never steady: it veers and backs, freshens and drops, and gusts rough up passing patches of
// water (cat's paws). The sea answers on many time scales: ripples turn within seconds, while swell raised by
// the old wind runs on for a minute or more (materials/oceanFFT.ts relaxes each wave toward the live wind).
//
// The wind here is a pure function of time, built from smooth seeded noise, so every reader (the FFT, both
// oceans, the readout) agrees on it without sharing state, and reloads and tests see the same weather.

export const WIND = {
  /** How far the direction wanders either side of its mean, radians. */
  veer: 0.6,
  /** Seconds per wander of the direction: a slow swing and a quicker shift on top. */
  veerTimes: [55, 13],
  /** The speed's slow lulls and freshening, as a share of the mean, and their timescale in seconds. */
  freshen: 0.22,
  freshenTime: 34,
  /** Gusts: extra speed at a gust's peak, as a share of the mean, and the timescale of one. */
  gust: 0.3,
  gustTime: 4.5,
  /** Seconds per swing of gustiness: how many cat's paws are about. */
  gustinessTime: 21,
  /** How fast a cat's paw sweeps downwind, orb radii per second per unit of wind speed. */
  gustTravel: 0.06,
  /** Wave periods a wave takes to grow toward a stronger wind, and to die back when it drops or turns. */
  growPeriods: 1.5,
  decayPeriods: 4,
  /** The surface current the wind drives, orb radii per second per unit of wind speed. */
  drift: 0.035,
} as const;

/** Tuned wind speed per Beaufort level: the default wind, 0.9, is a light breeze. */
export const BEAUFORT_STEP = 0.45;
/** The levels the wind control offers, Calm to Strong breeze; past that the orb's sea is all whitecaps. */
export const BEAUFORT_NAMES = [
  'Calm',
  'Light air',
  'Light breeze',
  'Gentle breeze',
  'Moderate breeze',
  'Fresh breeze',
  'Strong breeze',
] as const;

/** The tuned wind speed for a Beaufort level. */
export function windForBeaufort(level: number): number {
  return level * BEAUFORT_STEP;
}

/**
 * How much bigger the waves are at a tuned wind speed than at the default, by their slopes. A real sea grows
 * with the wind mostly by lengthening, not steepening: Cox and Munk measured the slope variance as
 * 0.003 + 0.00512 U (U in m/s), so from Beaufort 2 to 6 slopes only double. (The spectrum's height going as
 * speed^1.5 at a fixed peak wavelength would steepen it until every crest broke.)
 */
export function seaGain(speed: number): number {
  const slope = (s: number) => {
    const metres = 0.836 * Math.max(0, s / BEAUFORT_STEP) ** 1.5;
    return Math.sqrt(0.003 + 0.00512 * metres);
  };
  return slope(speed) / slope(OCEAN_DEFAULTS.windSpeed);
}

/** The Beaufort level, 0 to 12, of a wind speed plus the touch-made rough weather (0 to 1) on top of it. */
export function beaufortFor(speed: number, storm = 0): number {
  return Math.min(12, Math.max(0, Math.round(speed / BEAUFORT_STEP + storm * 5.6)));
}

export interface WindState {
  /** Direction the wind blows toward, in cube-face axes from east toward north (as OCEAN_WIND_ANGLE). */
  angle: number;
  /** Speed, in the units of OceanTuning.windSpeed. */
  speed: number;
  /** 0 to 1: how gusty it is right now, which sets how many cat's paws cross the sea. */
  gustiness: number;
}

/** Smooth seeded value noise in time, -1 to 1. */
function noise(t: number, seed: number): number {
  const i = Math.floor(t);
  const f = t - i;
  const hash = (n: number) => {
    let h = Math.imul(n ^ Math.imul(seed, 0x27d4eb2d), 0x9e3779b1);
    h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
    h ^= h >>> 13;
    return (h >>> 0) / 0xffffffff;
  };
  const s = f * f * (3 - 2 * f);
  return (hash(i) + (hash(i + 1) - hash(i)) * s) * 2 - 1;
}

/** The wind at `time` seconds, around a mean speed (the tuned wind). Calm stays calm. */
export function oceanWind(time: number, meanSpeed: number, seed = 1): WindState {
  const [slow, quick] = WIND.veerTimes;
  const angle =
    OCEAN_WIND_ANGLE +
    WIND.veer * (0.75 * noise(time / slow, seed) + 0.25 * noise(time / quick + 31, seed + 1));
  const gust = Math.max(0, noise(time / WIND.gustTime + 13, seed + 2)) ** 1.5;
  const factor = 1 + WIND.freshen * noise(time / WIND.freshenTime + 7, seed + 3) + WIND.gust * gust;
  const gustiness = Math.min(1, Math.max(0, 0.55 + 0.45 * noise(time / WIND.gustinessTime + 5, seed + 4)));
  return { angle, speed: Math.max(0, meanSpeed * factor), gustiness };
}

/**
 * One wave's amplitude gain relaxing toward its target for the live wind: quickly while the wind builds it,
 * slowly as it dies away, both in proportion to the wave's period. The GPU pass does the same.
 */
export function relaxGain(gain: number, target: number, dt: number, omega: number): number {
  const periods = target > gain ? WIND.growPeriods : WIND.decayPeriods;
  return target + (gain - target) * Math.exp((-dt * omega) / (2 * Math.PI * periods));
}

/**
 * The wind's direction on the orb at unit direction p: `angle` from east toward north, the axes the FFT's
 * equatorial cube faces use. Its length is the cosine of the latitude, so it dies away smoothly at the poles
 * rather than spinning there.
 */
export function windAt(p: Vector3, angle: number, out = new Vector3()): Vector3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  // east = up x p, north = up - p p.y
  return out.set(c * p.z - s * p.x * p.y, s * (1 - p.y * p.y), -c * p.x - s * p.z * p.y);
}

export const OCEAN_GUSTS = 6;

export interface Gust {
  readonly center: Vector3;
  age: number;
  life: number;
  /** 0 to 1 at its peak. */
  strength: number;
}

/** Cat's paws: patches of gusting wind roughening the water, sweeping downwind and fading. */
export class GustField {
  readonly gusts: Gust[];
  private seed: number;
  private readonly direction = new Vector3();
  private readonly axis = new Vector3();

  constructor(seed = 1) {
    this.seed = seed >>> 0 || 1;
    this.gusts = Array.from({ length: OCEAN_GUSTS }, (_, i) => {
      const gust: Gust = { center: new Vector3(), age: 0, life: 1, strength: 0 };
      this.respawn(gust);
      // start them at staggered points of their lives
      gust.age = gust.life * (i / OCEAN_GUSTS);
      return gust;
    });
  }

  /** Each gust's strength now, 0 to 1. */
  intensity(gust: Gust): number {
    return gust.strength * Math.sin(Math.PI * Math.min(1, gust.age / gust.life));
  }

  update(dt: number, wind: WindState): void {
    for (const gust of this.gusts) {
      gust.age += dt;
      if (gust.age >= gust.life) {
        this.respawn(gust);
        gust.strength *= wind.gustiness;
      }
      // carried downwind along a great circle
      windAt(gust.center, wind.angle, this.direction);
      if (this.direction.lengthSq() < 1e-6) continue;
      this.axis.crossVectors(gust.center, this.direction).normalize();
      gust.center.applyAxisAngle(this.axis, wind.speed * WIND.gustTravel * dt).normalize();
    }
  }

  private random(): number {
    this.seed ^= this.seed << 13;
    this.seed ^= this.seed >>> 17;
    this.seed ^= this.seed << 5;
    return (this.seed >>> 0) / 0x100000000;
  }

  private respawn(gust: Gust): void {
    const z = this.random() * 2 - 1;
    const a = this.random() * 2 * Math.PI;
    const r = Math.sqrt(1 - z * z);
    gust.center.set(r * Math.cos(a), z, r * Math.sin(a));
    gust.age = 0;
    gust.life = 6 + this.random() * 6;
    gust.strength = 0.5 + this.random() * 0.5;
  }
}
