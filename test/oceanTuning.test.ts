import { describe, expect, it } from 'vitest';
import { createOceanSpectrum } from '../src/core/oceanFFT';
import { OCEAN_DEFAULTS, oceanDetail, oceanTuningQuery, readOceanTuning } from '../src/core/oceanTuning';
import { createDeformGeometry } from '../src/materials/deform';

describe('ocean tuning', () => {
  it('uses the selected defaults, clamps URL overrides, and ignores obsolete LOD settings', () => {
    expect(readOceanTuning('')).toEqual({
      windSpeed: 0.9,
      peakWavelength: 0.23,
      amplitude: 0.95,
      choppiness: 2,
      foam: 1.4,
    });
    const tuning = readOceanTuning('?oceanWind=9&oceanWavelength=0.2&oceanFoam=nope&oceanLod=high');
    expect(tuning).toMatchObject({ windSpeed: 3, peakWavelength: 0.2, foam: 1.4 });
    expect(tuning).not.toHaveProperty('lod');
  });

  it('writes the tuning back into the URL, keeping other parameters and leaving out defaults', () => {
    expect(oceanTuningQuery(OCEAN_DEFAULTS, '?material=ocean&oceanWind=2')).toBe('?material=ocean');
    const windy = { ...OCEAN_DEFAULTS, windSpeed: 0.45 * 3 };
    const query = oceanTuningQuery(windy, '?material=coast');
    expect(query).toBe('?material=coast&oceanWind=1.35');
    expect(readOceanTuning(query)).toEqual(windy);
    expect(oceanTuningQuery(OCEAN_DEFAULTS)).toBe('');
  });

  it('uses about 8k vertices on mobile and 24k on desktop', () => {
    const counts = [true, false].map((coarse) => {
      const geometry = createDeformGeometry(true, oceanDetail(coarse));
      const count = geometry.getAttribute('position').count;
      geometry.dispose();
      return count;
    });
    expect(counts).toEqual([8412, 24012]);
  });

  it('lengthens the sea with the peak wavelength and calms it with the wind', () => {
    // energy-weighted mean wavenumber of the longest cascade
    const meanK = (tuning: typeof OCEAN_DEFAULTS) => {
      const { h0, waves, size } = createOceanSpectrum(true, tuning);
      let power = 0;
      let weighted = 0;
      for (let i = 0; i < size * size; i++) {
        const e = h0[i * 4]! ** 2 + h0[i * 4 + 1]! ** 2;
        power += e;
        weighted += e * Math.hypot(waves[i * 4]!, waves[i * 4 + 1]!);
      }
      return weighted / power;
    };
    expect(meanK({ ...OCEAN_DEFAULTS, peakWavelength: 0.6 })).toBeLessThan(meanK(OCEAN_DEFAULTS) * 0.7);
    const energy = (windSpeed: number) =>
      createOceanSpectrum(true, { ...OCEAN_DEFAULTS, windSpeed }).h0.reduce((sum, v) => sum + v * v, 0);
    expect(energy(0.45)).toBeLessThan(energy(0.9));
  });
});
