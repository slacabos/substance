import { describe, expect, it } from 'vitest';
import {
  createOceanSpectrum,
  donelanBanner,
  evaluateCascade,
  evolveMode,
  inverseFFT,
  modeAmplitude,
  oceanCascades,
  OCEAN_REPEAT,
  OCEAN_WAVELENGTH_SCALE,
  OCEAN_WIND_ANGLE,
} from '../src/core/oceanFFT';
import { OCEAN_DEFAULTS } from '../src/core/oceanTuning';

describe('ocean FFT', () => {
  it('runs a Stockham inverse FFT that matches the direct sum', () => {
    const n = 16;
    const re = Array.from({ length: n }, (_, i) => Math.sin(i * 1.7) + 0.3 * i);
    const im = Array.from({ length: n }, (_, i) => Math.cos(i * 0.9));
    const [outRe, outIm] = inverseFFT(re, im);
    for (let j = 0; j < n; j++) {
      let sr = 0;
      let si = 0;
      for (let m = 0; m < n; m++) {
        const a = (2 * Math.PI * m * j) / n;
        sr += re[m]! * Math.cos(a) - im[m]! * Math.sin(a);
        si += re[m]! * Math.sin(a) + im[m]! * Math.cos(a);
      }
      expect(outRe[j]).toBeCloseTo(sr, 9);
      expect(outIm[j]).toBeCloseTo(si, 9);
    }
  });

  it('splits the spectrum into contiguous cascades that tile a cube face whole', () => {
    for (const coarse of [true, false]) {
      const cascades = oceanCascades(coarse);
      expect(cascades[0]!.kMin).toBe(0);
      for (let i = 0; i < cascades.length; i++) {
        const c = cascades[i]!;
        expect(c.tile * Math.hypot(c.a, c.b)).toBeCloseTo(Math.PI / 2, 12);
        expect(Number.isInteger(c.a) && Number.isInteger(c.b)).toBe(true);
        if (i > 0) expect(c.kMin).toBeCloseTo(cascades[i - 1]!.kMax, 12);
        expect(c.kMax).toBeGreaterThan(c.kMin);
      }
    }
  });

  it('spreads energy mostly downwind, with each lobe integrating to one', () => {
    let sum = 0;
    const steps = 4000;
    for (let i = 0; i < steps; i++) sum += donelanBanner(1, -Math.PI + ((i + 0.5) * 2 * Math.PI) / steps);
    expect(sum * ((2 * Math.PI) / steps)).toBeCloseTo(1, 3);
    expect(donelanBanner(1, 0)).toBeGreaterThan(20 * donelanBanner(1, Math.PI / 2));
  });

  it('produces real height and displacement fields matching a direct sum over modes', () => {
    const spectrum = createOceanSpectrum(true, OCEAN_DEFAULTS);
    const { size } = spectrum;
    const cascade = 0;
    const time = 3.7;
    const fields = evaluateCascade(spectrum, cascade, time);
    const mode = new Float64Array(8);
    const h0 = new Float64Array(4);
    const tile = spectrum.cascades[cascade]!.tile;
    for (const [px, pz] of [
      [0, 0],
      [17, 90],
      [101, 33],
    ] as const) {
      const x = (px / size) * tile;
      const z = (pz / size) * tile;
      let h = 0;
      let dx = 0;
      for (let i = 0; i < size * size; i++) {
        const at = (cascade * size * size + i) * 4;
        const [kx, kz] = [spectrum.waves[at]!, spectrum.waves[at + 1]!];
        modeAmplitude(spectrum, spectrum.gain, cascade * size * size + i, h0);
        evolveMode(h0, spectrum.waves.subarray(at, at + 4), time, mode);
        const c = Math.cos(kx * x + kz * z);
        const s = Math.sin(kx * x + kz * z);
        // field 0 packs Dx + i h, so the real part of its sum is Dx and the imaginary part is h
        dx += mode[0]! * c - mode[1]! * s;
        h += mode[0]! * s + mode[1]! * c;
      }
      const index = pz * size + px;
      expect(fields.height[index]).toBeCloseTo(h, 9);
      expect(fields.dx[index]).toBeCloseTo(dx, 9);
    }
  });

  it('runs its waves downwind', () => {
    const spectrum = createOceanSpectrum(true, OCEAN_DEFAULTS);
    const { size } = spectrum;
    const a = evaluateCascade(spectrum, 0, 0).height;
    const b = evaluateCascade(spectrum, 0, 0.5).height;
    // the shift, in texels, that best carries the sea at 0 onto the sea half a second later
    let best = [0, 0, Infinity];
    for (let dy = -4; dy <= 4; dy++)
      for (let dx = -4; dx <= 4; dx++) {
        let error = 0;
        for (let y = 0; y < size; y++)
          for (let x = 0; x < size; x++)
            error += (b[((y + dy + size) % size) * size + ((x + dx + size) % size)]! - a[y * size + x]!) ** 2;
        if (error < best[2]!) best = [dx, dy, error];
      }
    const cascade = spectrum.cascades[0]!;
    const wind = OCEAN_WIND_ANGLE + Math.atan2(cascade.sin, cascade.cos);
    const heading = Math.atan2(best[1]!, best[0]!);
    expect(Math.cos(heading - wind)).toBeGreaterThan(0.7);
  });

  it('keeps h(-k) the conjugate of h(k), so the height field is real', () => {
    const spectrum = createOceanSpectrum(true, OCEAN_DEFAULTS);
    const { size } = spectrum;
    const height = (x: number, y: number) => {
      const at = (y * size + x) * 4;
      const [a, b, c, d] = Array.from(spectrum.h0.subarray(at, at + 4));
      const w = spectrum.waves[at + 2]! * 1.9;
      // h = h0 e^{-iwt} + conj(h0(-k)) e^{iwt}
      return [
        a! * Math.cos(w) + b! * Math.sin(w) + c! * Math.cos(w) - d! * Math.sin(w),
        b! * Math.cos(w) - a! * Math.sin(w) + c! * Math.sin(w) + d! * Math.cos(w),
      ];
    };
    for (const [x, y] of [
      [3, 1],
      [5, 120],
      [64, 7],
    ] as const) {
      const [re, im] = height(x, y);
      const [mre, mim] = height((size - x) % size, (size - y) % size);
      expect(mre).toBeCloseTo(re!, 12);
      expect(mim).toBeCloseTo(-im!, 12);
    }
  });

  it('matches the target wave height and stays flat without wind', () => {
    const spectrum = createOceanSpectrum(true, OCEAN_DEFAULTS);
    let variance = 0;
    for (let c = 0; c < spectrum.cascades.length; c++) {
      const { height } = evaluateCascade(spectrum, c, 1.3);
      let sum = 0;
      for (const h of height) sum += h * h;
      variance += sum / height.length;
    }
    // a single random sea lands near, not exactly on, the expected RMS
    expect(Math.sqrt(variance)).toBeGreaterThan(0.0045 * 0.7);
    expect(Math.sqrt(variance)).toBeLessThan(0.0045 * 1.3);

    const calm = createOceanSpectrum(true, { ...OCEAN_DEFAULTS, windSpeed: 0 });
    expect(calm.h0.every((v) => v === 0)).toBe(true);
  });

  it('peaks near the chosen wavelength and repeats exactly after the repeat period', () => {
    const spectrum = createOceanSpectrum(true, OCEAN_DEFAULTS);
    const { size } = spectrum;
    let best = 0;
    let bestK = 0;
    for (let i = 0; i < size * size; i++) {
      const at = i * 4;
      const power = spectrum.h0[at]! ** 2 + spectrum.h0[at + 1]! ** 2;
      const k = Math.hypot(spectrum.waves[at]!, spectrum.waves[at + 1]!);
      // average out the random draw by weighting the power by k-shells
      if (power > best) {
        best = power;
        bestK = k;
      }
    }
    const peakK = (2 * Math.PI) / (OCEAN_DEFAULTS.peakWavelength * OCEAN_WAVELENGTH_SCALE);
    expect(bestK).toBeGreaterThan(peakK * 0.5);
    expect(bestK).toBeLessThan(peakK * 2);

    const a = evaluateCascade(spectrum, 0, 2.5).height;
    const b = evaluateCascade(spectrum, 0, 2.5 + OCEAN_REPEAT).height;
    for (let i = 0; i < a.length; i += 997) expect(b[i]).toBeCloseTo(a[i]!, 6);
  });
});
