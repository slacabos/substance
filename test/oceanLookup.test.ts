import { describe, expect, it } from 'vitest';
import { buildDirectionLookup, lookupCell } from '../src/core/oceanLookup';
import { oceanMesh } from '../src/core/oceanMesh';
import { oceanDetail } from '../src/core/oceanTuning';
import { createDeformGeometry } from '../src/materials/deform';

describe('ocean direction lookup', () => {
  it('lands every direction on or next to its nearest vertex, across faces and seams', () => {
    const mesh = oceanMesh(createDeformGeometry(true, oceanDetail(false)));
    const lookup = buildDirectionLookup(mesh.positions);
    const p = mesh.positions;
    const count = p.length / 3;
    let seed = 3;
    const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
    const directions: number[][] = [
      [1, 1, 1],
      [-1, 0.999, 0],
      [0, 0, -1],
    ];
    for (let n = 0; n < 300; n++) directions.push([random(), random(), random()]);
    for (const [x, y, z] of directions) {
      const length = Math.hypot(x!, y!, z!);
      const [dx, dy, dz] = [x! / length, y! / length, z! / length];
      let nearest = 0;
      let best = Infinity;
      for (let i = 0; i < count; i++) {
        const d = (p[i * 3]! - dx) ** 2 + (p[i * 3 + 1]! - dy) ** 2 + (p[i * 3 + 2]! - dz) ** 2;
        if (d < best) [best, nearest] = [d, i];
      }
      const found = lookup[lookupCell(dx, dy, dz)]!;
      const candidates = [found, ...Array.from(mesh.neighbors.subarray(found * 6, found * 6 + 6))];
      expect(candidates).toContain(nearest);
    }
  });
});
