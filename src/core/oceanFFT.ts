import { OCEAN_FFT } from '../config';
import type { OceanTuning } from './oceanTuning';

// Tessendorf FFT ocean. The CPU side builds each cascade's initial spectrum and is the reference the GPU
// passes (materials/oceanFFT.ts) follow step for step. Lengths are in radians of the orb's surface.

/**
 * The sea is a finite-depth ocean, the same one the interactive waves (core/waveKernel.ts) move on: short waves
 * disperse like deep water, long ones run at sqrt(g H) like a tsunami. Depth and gravity are in orb radii; the
 * gravity keeps the peak swell at the period the hand-tuned waves had.
 */
export const OCEAN_DEPTH = 0.05;
export const OCEAN_GRAVITY = 0.06;
/** The peak-wavelength setting was authored against the old swell; this turns it into a real wavelength. */
export const OCEAN_WAVELENGTH_SCALE = (2 * Math.PI) / (13 * 0.33);
/** Mean wind direction in face space, measured from the eastward axis. */
export const OCEAN_WIND_ANGLE = 0.5;
/** Every wave's frequency is rounded so the sea repeats after this many seconds, keeping phases precise. */
export const OCEAN_REPEAT = 240;
/**
 * Height RMS at the default amplitude and wind. With the default wavelength that's a moderate wind sea whose
 * slopes stay near what Cox and Munk measured on the real ocean (RMS about 0.2).
 */
const BASE_RMS = 0.0045;
const REFERENCE_WIND = 0.9;
/** Shortest wave any cascade carries; finer ripples only show up as roughness. */
const K_MAX = (2 * Math.PI) / 0.005;
const QUARTER = Math.PI / 2;

export interface OceanCascade {
  /** Tile side, in radians. */
  tile: number;
  /** Lattice step per cube face; cos and sin of the tile's turn. */
  a: number;
  b: number;
  cos: number;
  sin: number;
  /** This cascade carries wavenumbers in (kMin, kMax]. */
  kMin: number;
  kMax: number;
  /** Half-width of the blend into the polar faces, as a ratio of axis components. */
  band: number;
}

export function oceanCascades(coarse: boolean): OceanCascade[] {
  const size = OCEAN_FFT.size(coarse);
  const lattice = OCEAN_FFT.cascades(coarse);
  const tiles = lattice.map(([a, b]) => QUARTER / Math.hypot(a, b));
  // Each cascade starts three tile modes above its own fundamental and ends where the next one starts.
  const start = (tile: number) => (6 * Math.PI) / tile;
  return lattice.map(([a, b], i) => {
    const tile = tiles[i]!;
    const next = tiles[i + 1];
    const s = Math.hypot(a, b);
    return {
      tile,
      a,
      b,
      cos: a / s,
      sin: b / s,
      kMin: i === 0 ? 0 : start(tile),
      kMax: next ? start(next) : Math.min(K_MAX, (0.8 * Math.PI * size) / tile),
      band: 0.3 / Math.sqrt(s),
    };
  });
}

/** Finite-depth dispersion, w^2 = g k tanh(k H). */
export function dispersion(k: number): number {
  return Math.sqrt(OCEAN_GRAVITY * k * Math.tanh(k * OCEAN_DEPTH));
}

/** d omega / dk, which turns the frequency spectrum into a wavenumber one. */
function groupSpeed(k: number): number {
  const kh = k * OCEAN_DEPTH;
  const sech = 1 / Math.cosh(kh);
  return (OCEAN_GRAVITY * (Math.tanh(kh) + kh * sech * sech)) / (2 * dispersion(k));
}

/** Dispersion rounded to a whole number of cycles per repeat. */
export function oceanOmega(k: number): number {
  const step = (2 * Math.PI) / OCEAN_REPEAT;
  return Math.floor(dispersion(k) / step) * step;
}

/** JONSWAP frequency spectrum, unnormalised. */
export function jonswap(omega: number, peak: number, gamma = 3.3): number {
  if (omega <= 0) return 0;
  const sigma = omega <= peak ? 0.07 : 0.09;
  const r = Math.exp(-((omega - peak) ** 2) / (2 * sigma * sigma * peak * peak));
  return omega ** -5 * Math.exp(-1.25 * (peak / omega) ** 4) * gamma ** r;
}

/** Donelan–Banner directional spreading; theta is from the wind, and it integrates to 1 over a turn. */
export function donelanBanner(ratio: number, theta: number): number {
  const beta =
    ratio < 0.95
      ? 2.61 * ratio ** 1.3
      : ratio < 1.6
        ? 2.28 * ratio ** -0.65
        : 10 ** (-0.4 + 0.8393 * Math.exp(-0.567 * Math.log(ratio * ratio)));
  const t = Math.atan2(Math.sin(theta), Math.cos(theta));
  const sech = 1 / Math.cosh(beta * t);
  return (beta / (2 * Math.tanh(beta * Math.PI))) * sech * sech;
}

export interface OceanSpectrum {
  size: number;
  cascades: OceanCascade[];
  /** Per cascade and mode, vec4: h0(k) and conj(h0(-k)), real and imaginary parts. */
  h0: Float32Array;
  /** Per cascade and mode, vec4: kx, kz, omega, 1 / |k|. */
  waves: Float32Array;
  /** Expected slope variance (both directions) each cascade carries, before amplitude. */
  slopeVariance: number[];
}

/** Seeded normal deviates (xorshift + Box–Muller), so every load shows the same sea. */
function gaussian(seed: number): () => number {
  let s = seed >>> 0 || 1;
  const next = () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) + 0.5) / 0x100000000;
  };
  return () => Math.sqrt(-2 * Math.log(next())) * Math.cos(2 * Math.PI * next());
}

/** Signed frequency index for FFT order: 0..n/2-1, then -n/2..-1. */
const frequency = (m: number, n: number) => (m < n / 2 ? m : m - n);

/** Each cascade's starting spectrum, normalised so the sea's height RMS follows wind (amplitude is applied later). */
export function createOceanSpectrum(coarse: boolean, tuning: Readonly<OceanTuning>, seed = 1): OceanSpectrum {
  const size = OCEAN_FFT.size(coarse);
  const cascades = oceanCascades(coarse);
  const modes = size * size;
  const energy = new Float64Array(cascades.length * modes);
  const waves = new Float32Array(cascades.length * modes * 4);
  const peakK = (2 * Math.PI) / (tuning.peakWavelength * OCEAN_WAVELENGTH_SCALE);
  const peakOmega = dispersion(peakK);

  let total = 0;
  cascades.forEach((cascade, c) => {
    const dk = (2 * Math.PI) / cascade.tile;
    // the tile is turned, so the wind is too, to keep it the same on the orb
    const wind = OCEAN_WIND_ANGLE + Math.atan2(cascade.sin, cascade.cos);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = c * modes + y * size + x;
        const kx = frequency(x, size) * dk;
        const kz = frequency(y, size) * dk;
        const k = Math.hypot(kx, kz);
        waves.set([kx, kz, oceanOmega(k), k > 0 ? 1 / k : 0], i * 4);
        if (k <= cascade.kMin || k > cascade.kMax) continue;
        const omega = dispersion(k);
        // E(kx, kz) dkx dkz = S(omega) D(theta) domega dtheta, with dkx dkz = k dk dtheta
        const density =
          (jonswap(omega, peakOmega) *
            donelanBanner(omega / peakOmega, Math.atan2(kz, kx) - wind) *
            groupSpeed(k)) /
          k;
        energy[i] = density * dk * dk;
        total += energy[i]!;
      }
    }
  });

  // The real field's variance is twice the sum, since h(k) and h(-k) both carry each mode.
  const rms = BASE_RMS * Math.pow(Math.max(0, tuning.windSpeed) / REFERENCE_WIND, 1.5);
  const scale = total > 0 ? (rms * rms) / (2 * total) : 0;
  const random = gaussian(seed);
  const h0 = new Float32Array(cascades.length * modes * 4);
  for (let i = 0; i < energy.length; i++) {
    const amplitude = Math.sqrt((energy[i]! * scale) / 2);
    const re = random() * amplitude;
    const im = random() * amplitude;
    h0[i * 4] = re;
    h0[i * 4 + 1] = im;
  }
  const slopeVariance = cascades.map((_, c) => {
    let sum = 0;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = c * modes + y * size + x;
        const mirror = c * modes + ((size - y) % size) * size + ((size - x) % size);
        // conj(h0(-k))
        h0[i * 4 + 2] = h0[mirror * 4]!;
        h0[i * 4 + 3] = -h0[mirror * 4 + 1]!;
        const k2 = waves[i * 4]! ** 2 + waves[i * 4 + 1]! ** 2;
        sum += 2 * energy[i]! * scale * k2;
      }
    }
    return sum;
  });
  return { size, cascades, h0, waves, slopeVariance };
}

/** The four complex fields each IFFT carries, packed two real fields apiece. */
export const OCEAN_FIELDS = 4;

/**
 * Spectrum of one mode at time t, packed for the inverse FFT as
 * [Dx + i h, Dz + i dDx/dz, dh/dx + i dh/dz, dDx/dx + i dDz/dz]: four complex numbers, real part first.
 */
export function evolveMode(
  h0: ArrayLike<number>,
  wave: ArrayLike<number>,
  time: number,
  out: Float64Array,
): void {
  const [kx, kz, omega, invK] = [wave[0]!, wave[1]!, wave[2]!, wave[3]!];
  const c = Math.cos(omega * time);
  const s = Math.sin(omega * time);
  // h = h0 e^{i w t} + conj(h0(-k)) e^{-i w t}
  const re = h0[0]! * c - h0[1]! * s + h0[2]! * c + h0[3]! * s;
  const im = h0[0]! * s + h0[1]! * c - h0[2]! * s + h0[3]! * c;
  const ux = kx * invK;
  const uz = kz * invK;
  // Dx = -i (kx/k) h,  i h = (-im, re)
  out[0] = ux * im - im;
  out[1] = -ux * re + re;
  // Dz = -i (kz/k) h,  i Dxz = i (kx kz / k) h
  out[2] = uz * im - kx * uz * im;
  out[3] = -uz * re + kx * uz * re;
  // dh/dx = i kx h,  i dh/dz = i (i kz h) = -kz h
  out[4] = -kx * im - kz * re;
  out[5] = kx * re - kz * im;
  // dDx/dx = (kx^2/k) h,  i dDz/dz = i (kz^2/k) h
  out[6] = kx * ux * re - kz * uz * im;
  out[7] = kx * ux * im + kz * uz * re;
}

/**
 * One radix-2 Stockham stage of an inverse FFT along n points (out of place, natural order after log2 n
 * stages). p is 2^stage. The GPU stage kernel is this loop body.
 */
export function stockhamStage(
  re: ArrayLike<number>,
  im: ArrayLike<number>,
  outRe: Float64Array,
  outIm: Float64Array,
  n: number,
  p: number,
): void {
  const half = n / 2;
  for (let t = 0; t < half; t++) {
    const k = t & (p - 1);
    const angle = (Math.PI * k) / p;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const xr = re[t + half]!;
    const xi = im[t + half]!;
    const ur = xr * c - xi * s;
    const ui = xr * s + xi * c;
    const o = 2 * t - k;
    outRe[o] = re[t]! + ur;
    outIm[o] = im[t]! + ui;
    outRe[o + p] = re[t]! - ur;
    outIm[o + p] = im[t]! - ui;
  }
}

/** Unnormalised inverse FFT, sum over m of x[m] e^{2 pi i m j / n}. */
export function inverseFFT(re: ArrayLike<number>, im: ArrayLike<number>): [Float64Array, Float64Array] {
  const n = re.length;
  let a: [Float64Array, Float64Array] = [Float64Array.from(re), Float64Array.from(im)];
  let b: [Float64Array, Float64Array] = [new Float64Array(n), new Float64Array(n)];
  for (let p = 1; p < n; p *= 2) {
    stockhamStage(a[0], a[1], b[0], b[1], n, p);
    [a, b] = [b, a];
  }
  return a;
}

/**
 * The fields of one cascade at time t on its size² grid, from the CPU path: displacement x and z, height,
 * slopes and the displacement derivatives, each size² long and indexed y * size + x.
 */
export function evaluateCascade(spectrum: OceanSpectrum, cascade: number, time: number) {
  const { size } = spectrum;
  const modes = size * size;
  const fields = Array.from({ length: OCEAN_FIELDS }, (): [Float64Array, Float64Array] => [
    new Float64Array(modes),
    new Float64Array(modes),
  ]);
  const mode = new Float64Array(OCEAN_FIELDS * 2);
  for (let i = 0; i < modes; i++) {
    const at = (cascade * modes + i) * 4;
    evolveMode(spectrum.h0.subarray(at, at + 4), spectrum.waves.subarray(at, at + 4), time, mode);
    for (let f = 0; f < OCEAN_FIELDS; f++) {
      fields[f]![0][i] = mode[f * 2]!;
      fields[f]![1][i] = mode[f * 2 + 1]!;
    }
  }
  for (const [re, im] of fields) {
    // rows, then columns
    for (let pass = 0; pass < 2; pass++) {
      for (let line = 0; line < size; line++) {
        const index = (j: number) => (pass === 0 ? line * size + j : j * size + line);
        const lineRe = new Float64Array(size);
        const lineIm = new Float64Array(size);
        for (let j = 0; j < size; j++) {
          lineRe[j] = re[index(j)]!;
          lineIm[j] = im[index(j)]!;
        }
        const [outRe, outIm] = inverseFFT(lineRe, lineIm);
        for (let j = 0; j < size; j++) {
          re[index(j)] = outRe[j]!;
          im[index(j)] = outIm[j]!;
        }
      }
    }
  }
  const [f0, f1, f2, f3] = fields as [
    (typeof fields)[0],
    (typeof fields)[0],
    (typeof fields)[0],
    (typeof fields)[0],
  ];
  return {
    dx: f0[0],
    height: f0[1],
    dz: f1[0],
    dxz: f1[1],
    slopeX: f2[0],
    slopeZ: f2[1],
    dxx: f3[0],
    dzz: f3[1],
  };
}
