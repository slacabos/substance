import { IcosahedronGeometry, Vector3 } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { describe, expect, it } from 'vitest';
import { OceanSim } from '../src/materials/oceanSim';

describe('ocean surface simulation', () => {
  it('turns a tap into height and foam, then clears on reset', () => {
    const geometry = new IcosahedronGeometry(1, 5);
    geometry.deleteAttribute('uv');
    const indexed = mergeVertices(geometry);
    const sim = new OceanSim(indexed);
    const p = indexed.getAttribute('position');
    const dir = new Vector3(p.getX(0), p.getY(0), p.getZ(0)).normalize();
    sim.update(1 / 60, [{ dir, strength: -1.15 }]);
    const heights = indexed.getAttribute('oceanHeight');
    const foam = indexed.getAttribute('oceanFoam');
    expect(Math.abs(heights.getX(0))).toBeGreaterThan(0.001);
    expect(foam.getX(0)).toBeGreaterThan(0.05);
    expect(Math.abs(sim.sampleAt(dir).height)).toBeGreaterThan(0.001);
    sim.depositFoam(dir, 0.4);
    expect(sim.sampleAt(dir).foam).toBeGreaterThan(0.05);
    expect(sim.whitewaterSites(8).length).toBeGreaterThan(0);
    const affected = () => {
      let count = 0;
      for (let i = 0; i < heights.count; i++) if (Math.abs(heights.getX(i)) > 1e-5) count++;
      return count;
    };
    const initialAffected = affected();
    for (let i = 0; i < 30; i++) sim.update(1 / 60, []);
    expect(affected()).toBeGreaterThan(initialAffected);
    expect(Math.abs(heights.getX(0))).toBeGreaterThan(0);
    sim.reset();
    expect(heights.getX(0)).toBe(0);
    expect(foam.getX(0)).toBe(0);
  });

  it('advects deposited foam and lets it fade after contact ends', () => {
    const source = new IcosahedronGeometry(1, 5);
    source.deleteAttribute('uv');
    const sim = new OceanSim(mergeVertices(source));
    const dir = new Vector3(0, 0, 1);
    sim.depositFoam(dir, 0.8, 0.1);
    const initial = sim.sampleAt(dir).foam;
    for (let i = 0; i < 180; i++) sim.update(1 / 60, []);
    expect(sim.sampleAt(dir).foam).toBeLessThan(initial * 0.5);
  });
});
