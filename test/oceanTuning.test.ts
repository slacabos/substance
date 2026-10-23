import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { sampleOceanSurface } from '../src/core/oceanSpectrum';
import { OCEAN_DEFAULTS, oceanDetail, oceanTuningQuery, readOceanTuning } from '../src/core/oceanTuning';
import { createDeformGeometry } from '../src/materials/deform';

describe('ocean tuning', () => {
  it('clamps URL values, ignores invalid values, and preserves scene parameters', () => {
    const tuning = readOceanTuning('?oceanWind=9&oceanWavelength=0.2&oceanFoam=nope&oceanLod=high');
    expect(tuning).toMatchObject({ windSpeed: 3, peakWavelength: 0.2, foam: 1.3, lod: 'high' });
    const query = oceanTuningQuery('?material=ocean&shape=orb&oceanFoam=2', tuning);
    expect(new URLSearchParams(query).get('material')).toBe('ocean');
    expect(new URLSearchParams(query).get('shape')).toBe('orb');
    expect(new URLSearchParams(query).get('oceanFoam')).toBeNull();
    expect(readOceanTuning(query)).toEqual(tuning);
  });

  it('increases surface detail without changing the orb radius', () => {
    const counts = (['low', 'medium', 'high'] as const).map((lod) => {
      const geometry = createDeformGeometry(true, oceanDetail(lod));
      const count = geometry.getAttribute('position').count;
      geometry.dispose();
      return count;
    });
    expect(counts).toEqual([8412, 20252, 53292]);
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
