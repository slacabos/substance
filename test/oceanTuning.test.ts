import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { sampleOceanSurface } from '../src/core/oceanSpectrum';
import { OCEAN_DEFAULTS, oceanDetail, readOceanTuning } from '../src/core/oceanTuning';
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

  it('uses about 8k vertices on mobile and 24k on desktop', () => {
    const counts = [true, false].map((coarse) => {
      const geometry = createDeformGeometry(true, oceanDetail(coarse));
      const count = geometry.getAttribute('position').count;
      geometry.dispose();
      return count;
    });
    expect(counts).toEqual([8412, 24012]);
  });

  it('moves the same spherical surface for amplitude, wavelength, and choppiness', () => {
    const dir = new Vector3(0.42, 0.8, 0.42).normalize();
    const a = new Vector3();
    const b = new Vector3();
    const base = sampleOceanSurface(dir, 0.7, 0, 0, a);
    expect(base).not.toBe(0);
    expect(sampleOceanSurface(dir, 0.7, 0, 0, b, { ...OCEAN_DEFAULTS, amplitude: 0 })).toBe(0);
    expect(b.distanceTo(dir)).toBeLessThan(1e-6);
    const longWave = sampleOceanSurface(dir, 0.7, 0, 0, b, { ...OCEAN_DEFAULTS, peakWavelength: 0.6 });
    expect(longWave).not.toBeCloseTo(base, 5);
    sampleOceanSurface(dir, 0.7, 0, 0, b, { ...OCEAN_DEFAULTS, choppiness: 0 });
    expect(a.distanceTo(b)).toBeGreaterThan(1e-4);
  });
});
