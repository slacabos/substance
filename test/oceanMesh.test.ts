import { IcosahedronGeometry } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { describe, expect, it } from 'vitest';
import { oceanMesh } from '../src/core/oceanMesh';

describe('ocean mesh adjacency', () => {
  it('covers every vertex with reciprocal neighbors and a finite spacing', () => {
    const geometry = new IcosahedronGeometry(1, 5);
    geometry.deleteAttribute('uv');
    const mesh = oceanMesh(mergeVertices(geometry));
    const count = mesh.positions.length / 3;
    expect(mesh.neighbors).toHaveLength(count * 6);
    expect(mesh.spacing).toBeGreaterThan(0);
    for (let i = 0; i < count; i++) {
      for (let k = 0; k < 6; k++) {
        const j = mesh.neighbors[i * 6 + k]!;
        expect(j).toBeLessThan(count);
        if (j !== i) expect([...mesh.neighbors.slice(j * 6, j * 6 + 6)]).toContain(i);
      }
    }
  });
});
