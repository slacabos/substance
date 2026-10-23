import { IcosahedronGeometry, Vector3 } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { describe, expect, it } from 'vitest';
import { OceanSim, oceanReleaseImpulse } from '../src/materials/oceanSim';

describe('ocean surface simulation', () => {
  it('turns a tap into a spreading height wave without immediate foam, then clears on reset', () => {
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
    expect(foam.getX(0)).toBe(0);
    expect(sim.whitewaterSites(8)).toHaveLength(0);
    expect(Math.abs(sim.sampleAt(dir).height)).toBeGreaterThan(0.001);
    sim.depositFoam(dir, 0.4);
    expect(sim.sampleAt(dir).foam).toBeGreaterThan(0.05);
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

  it('stores more wave energy during a long press than a short tap', () => {
    const run = (holdSeconds: number) => {
      const source = new IcosahedronGeometry(1, 28);
      source.deleteAttribute('uv');
      const geometry = mergeVertices(source);
      const sim = new OceanSim(geometry);
      const p = geometry.getAttribute('position');
      const dir = new Vector3(p.getX(0), p.getY(0), p.getZ(0)).normalize();
      sim.update(1 / 60, [{ dir, strength: -0.55 }]);
      for (let t = 1 / 60; t <= holdSeconds; t += 1 / 60) {
        sim.setPressure(dir, t);
        sim.update(1 / 60, []);
      }
      const heldDepth = Math.abs(sim.sampleAt(dir).height);
      sim.setPressure(null, 0);
      sim.update(1 / 60, [{ dir, strength: oceanReleaseImpulse(holdSeconds) }]);
      const heights = geometry.getAttribute('oceanHeight');
      let energy = 0;
      for (let i = 0; i < heights.count; i++) energy += heights.getX(i) ** 2;
      return { energy, heldDepth };
    };
    const short = run(0.08);
    const long = run(0.8);
    expect(long.heldDepth).toBeGreaterThan(short.heldDepth);
    expect(long.energy).toBeGreaterThan(short.energy * 1.5);
  });
});
