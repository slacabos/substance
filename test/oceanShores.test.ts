import { IcosahedronGeometry, Vector3 } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { describe, expect, it } from 'vitest';
import { SHORE_SITES, shoreDistance, shoreElevation, sampleShoreSurf } from '../src/core/oceanShores';
import { OceanShores } from '../src/materials/oceanShores';
import { OceanSim } from '../src/materials/oceanSim';

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

  it('generates breaking foam along the water side while keeping land dry', () => {
    const source = new IcosahedronGeometry(1, 18);
    source.deleteAttribute('uv');
    const geometry = mergeVertices(source);
    const fullIndexCount = geometry.getIndex()!.count;
    const sim = new OceanSim(geometry, true);
    expect(Object.keys(geometry.attributes).length).toBeLessThanOrEqual(8);
    expect(geometry.getAttribute('oceanShoreData').itemSize).toBe(4);
    expect(geometry.getIndex()!.count).toBeLessThan(fullIndexCount);
    expect(geometry.getIndex()!.count).toBeGreaterThan(fullIndexCount * 0.75);
    for (let i = 0; i < 60; i++) sim.update(1 / 30, [], (i + 1) / 30);
    const position = geometry.getAttribute('position');
    const foam = geometry.getAttribute('oceanFoam');
    const height = geometry.getAttribute('oceanHeight');
    const dir = new Vector3();
    let nearFoam = 0;
    let nearCount = 0;
    let dryCount = 0;
    let movingSurf = 0;
    for (let i = 0; i < position.count; i++) {
      dir.set(position.getX(i), position.getY(i), position.getZ(i));
      const distance = shoreDistance(dir);
      if (distance <= 0) {
        expect(foam.getX(i)).toBe(0);
        expect(height.getX(i)).toBe(0);
        dryCount++;
      } else if (distance < 0.075) {
        nearFoam += foam.getX(i);
        nearCount++;
        movingSurf = Math.max(movingSurf, Math.abs(height.getX(i)));
      }
    }
    expect(dryCount).toBeGreaterThan(0);
    expect(nearCount).toBeGreaterThan(0);
    expect(nearFoam / nearCount).toBeGreaterThan(0.02);
    expect(movingSurf).toBeGreaterThan(0.01);
    expect(sim.groundHeight(SHORE_SITES[0]!.center)).toBeGreaterThan(0.1);
    expect(sim.whitewaterSites(8).length).toBeGreaterThan(0);
  });
});
