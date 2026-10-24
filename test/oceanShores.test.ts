import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import {
  sampleShoreSurf,
  shelfLag,
  SHORE_SITES,
  shoreDistance,
  shoreElevation,
  shoreSurfPhase,
  shoreSwell,
} from '../src/core/oceanShores';
import { OCEAN_DEFAULTS } from '../src/core/oceanTuning';
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

  const swell = shoreSwell(OCEAN_DEFAULTS.peakWavelength);
  const surfAt = (distance: number, time: number, facing = 1) => {
    const out = { height: 0, breaking: 0 };
    // a straight shore: moving out from it runs against the swell by `facing`
    const phase = shoreSurfPhase(
      -distance * facing,
      distance,
      shelfLag(distance),
      facing,
      swell,
      swell.omega * time,
    );
    sampleShoreSurf(phase, distance, facing, 1, out);
    return out;
  };

  it('runs swell in toward the bank, crowding it on the shelf, and breaks it there', () => {
    // a crest is a point of fixed phase; the phase falls away from the shore, so find it by bisection
    const phaseAt = (d: number, time: number) =>
      shoreSurfPhase(-d, d, shelfLag(d), 1, swell, swell.omega * time);
    const crestAt = (phase: number, time: number) => {
      let [near, far] = [0, 0.1];
      for (let i = 0; i < 60; i++) {
        const mid = (near + far) / 2;
        if (phaseAt(mid, time) > phase) near = mid;
        else far = mid;
      }
      return near;
    };
    const target = phaseAt(0.05, 0);
    expect(crestAt(target, 1)).toBeLessThan(0.05 - 0.01);
    // and it slows as it comes in
    const outer = 0.05 - crestAt(target, 1);
    const inner = crestAt(phaseAt(0.01, 0), 0) - crestAt(phaseAt(0.01, 0), 1);
    expect(inner).toBeLessThan(outer);
    // a crest gains phase faster per unit distance near the bank than out at sea: it has slowed and shortened
    const slope = (d: number) =>
      Math.abs(
        shoreSurfPhase(-d - 1e-4, d + 1e-4, shelfLag(d + 1e-4), 1, swell, 0) -
          shoreSurfPhase(-d, d, shelfLag(d), 1, swell, 0),
      ) / 1e-4;
    expect(slope(0.005)).toBeGreaterThan(2 * slope(0.08));
    expect(slope(0.08)).toBeCloseTo(swell.k, 3);
    let breaking = 0;
    for (let time = 0; time < 10; time += 0.05) breaking = Math.max(breaking, surfAt(0.012, time).breaking);
    expect(breaking).toBeGreaterThan(0.9);
    expect(surfAt(-0.01, 3)).toEqual({ height: 0, breaking: 0 });
    expect(surfAt(0.2, 3)).toEqual({ height: 0, breaking: 0 });
  });

  it('breaks hardest on shores facing the swell, and keeps a steady pace however long it runs', () => {
    let exposed = 0;
    let lee = 0;
    for (let time = 0; time < 10; time += 0.05) {
      exposed = Math.max(exposed, surfAt(0.012, time, 1).breaking);
      lee = Math.max(lee, surfAt(0.012, time, -1).breaking);
    }
    expect(lee).toBeLessThan(0.25 * exposed);
    // the clock only ever adds omega per second, never time times a changing speed
    for (const time of [1, 100, 10000]) {
      const step =
        shoreSurfPhase(0, 0.01, shelfLag(0.01), 1, swell, swell.omega * (time + 1)) -
        shoreSurfPhase(0, 0.01, shelfLag(0.01), 1, swell, swell.omega * time);
      expect(step).toBeCloseTo(-swell.omega, 6);
    }
  });
});
