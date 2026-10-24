import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { SHORE_SITES, shoreDistance, shoreElevation, sampleShoreSurf } from '../src/core/oceanShores';
import { OceanShores } from '../src/materials/oceanShores';

describe('coastal ocean', () => {
  it('raises inland terrain above the largest ambient swells', () => {
    expect(SHORE_SITES.length).toBeGreaterThanOrEqual(4);
    for (const site of SHORE_SITES) {
      expect(shoreDistance(site.center)).toBeLessThan(0);
      expect(shoreElevation(site, site.center)).toBeGreaterThan(0.1);
    }
    expect(shoreDistance(new Vector3(0, 0, -1))).toBeGreaterThan(0);
    const shores = new OceanShores(true);
    const position = shores.mesh.geometry.getAttribute('position');
    const before = position.getX(0);
    shores.updateMorph(1);
    expect(position.getX(0)).not.toBe(before);
  });

  it('moves a displaced crest toward the bank and breaks it there', () => {
    const out = { height: 0, breaking: 0 };
    const peak = (time: number) => {
      let height = -Infinity;
      let distance = 0;
      for (let d = 0.065; d < 0.155; d += 0.001) {
        sampleShoreSurf(d, 0, time, 1, 1, 1, out);
        if (out.height > height) {
          height = out.height;
          distance = d;
        }
      }
      return { height, distance };
    };
    const first = peak(1);
    const later = peak(1.2);
    expect(first.height).toBeGreaterThan(0.02);
    expect(later.distance).toBeLessThan(first.distance - 0.008);
    const impactTime = (Math.PI / 2 + Math.PI * 2 - 0.015 * 43) / 3.1;
    sampleShoreSurf(0.015, 0, impactTime, 1, 1, 1, out);
    expect(out.breaking).toBeGreaterThan(0.9);
    sampleShoreSurf(-0.01, 0, impactTime, 1, 1, 1, out);
    expect(out.height).toBe(0);
    expect(out.breaking).toBe(0);
  });
});
