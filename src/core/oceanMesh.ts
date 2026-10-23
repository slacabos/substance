import type { BufferGeometry } from 'three';

/** Six adjacent vertices per point of the indexed icosphere; degree-five vertices repeat themselves. */
export function oceanMesh(geometry: BufferGeometry): {
  positions: Float32Array;
  neighbors: Uint32Array;
  spacing: number;
} {
  const index = geometry.getIndex();
  const attribute = geometry.getAttribute('position');
  if (!index || !attribute) throw new Error('Ocean simulation needs indexed positions');

  const count = attribute.count;
  const positions = new Float32Array(count * 3);
  const adjacent = Array.from({ length: count }, () => new Set<number>());
  for (let i = 0; i < count; i++) {
    positions[i * 3] = attribute.getX(i);
    positions[i * 3 + 1] = attribute.getY(i);
    positions[i * 3 + 2] = attribute.getZ(i);
  }
  const add = (a: number, b: number): void => {
    adjacent[a]!.add(b);
    adjacent[b]!.add(a);
  };
  for (let i = 0; i < index.count; i += 3) {
    const a = index.getX(i);
    const b = index.getX(i + 1);
    const c = index.getX(i + 2);
    add(a, b);
    add(b, c);
    add(c, a);
  }

  const neighbors = new Uint32Array(count * 6);
  let distance = 0;
  let edges = 0;
  for (let i = 0; i < count; i++) {
    const ring = [...adjacent[i]!];
    if (ring.length < 5 || ring.length > 6) throw new Error(`Unexpected ocean vertex degree ${ring.length}`);
    for (let j = 0; j < 6; j++) neighbors[i * 6 + j] = ring[j] ?? i;
    for (const j of ring) {
      const dx = positions[i * 3]! - positions[j * 3]!;
      const dy = positions[i * 3 + 1]! - positions[j * 3 + 1]!;
      const dz = positions[i * 3 + 2]! - positions[j * 3 + 2]!;
      distance += Math.hypot(dx, dy, dz);
      edges++;
    }
  }
  return { positions, neighbors, spacing: distance / edges };
}
