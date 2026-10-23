import { IcosahedronGeometry, Vector3 } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { describe, expect, it } from 'vitest';
import { OceanSim } from '../src/materials/oceanSim';
import { OceanSpray } from '../src/materials/oceanSpray';

describe('ocean spray', () => {
  it('returns to the surface and leaves foam', () => {
    const source = new IcosahedronGeometry(1, 18);
    source.deleteAttribute('uv');
    const sim = new OceanSim(mergeVertices(source));
    const spray = new OceanSpray(true);
    const touch = new Vector3(0, 0, 1);
    spray.emitCrest(touch, new Vector3(0, 1, 0), 1, 30);
    for (let i = 0; i < 90; i++) spray.update(1 / 60, sim);
    let deposited = 0;
    for (let y = -5; y <= 10; y++)
      for (let x = -5; x <= 5; x++)
        deposited = Math.max(deposited, sim.sampleAt(new Vector3(x * 0.04, y * 0.04, 1).normalize()).foam);
    expect(deposited).toBeGreaterThan(0);
    spray.clear();
  });
});
