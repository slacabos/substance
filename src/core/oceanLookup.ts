// Finding the ocean vertex nearest a direction, on the GPU: a cube map of nearest-vertex indices. A direction
// picks its cube face by its largest component, then a cell by its other two components divided by that one.
// The GPU reads the cell's vertex and blends it with that vertex's six neighbours.

/** Cells per side of each cube face. */
export const LOOKUP_SIZE = 96;

/** For each face (+x, -x, +y, -y, +z, -z), the two axes its cell coordinates run along. */
export const LOOKUP_AXES: readonly (readonly [number, number])[] = [
  [1, 2],
  [1, 2],
  [0, 2],
  [0, 2],
  [0, 1],
  [0, 1],
];

/** The lookup cell of a direction (need not be unit length). */
export function lookupCell(x: number, y: number, z: number, size = LOOKUP_SIZE): number {
  const d = [x, y, z];
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const az = Math.abs(z);
  const axis = ax >= ay && ax >= az ? 0 : ay >= az ? 1 : 2;
  const face = axis * 2 + (d[axis]! >= 0 ? 0 : 1);
  const [ua, va] = LOOKUP_AXES[face]!;
  const major = Math.abs(d[axis]!);
  const cell = (v: number) => Math.min(size - 1, Math.max(0, Math.floor(((v / major + 1) / 2) * size)));
  return face * size * size + cell(d[va]!) * size + cell(d[ua]!);
}

/** Nearest vertex (by chord distance) for every cell's centre direction. */
export function buildDirectionLookup(positions: Float32Array, size = LOOKUP_SIZE): Uint32Array {
  const count = positions.length / 3;
  // bucket the vertices in a coarse 3D grid so each query only looks nearby
  const bins = 48;
  const bin = (v: number) => Math.min(bins - 1, Math.max(0, Math.floor(((v + 1.05) / 2.1) * bins)));
  const buckets = new Map<number, number[]>();
  for (let i = 0; i < count; i++) {
    const key =
      (bin(positions[i * 3]!) * bins + bin(positions[i * 3 + 1]!)) * bins + bin(positions[i * 3 + 2]!);
    let list = buckets.get(key);
    if (!list) buckets.set(key, (list = []));
    list.push(i);
  }
  const lookup = new Uint32Array(6 * size * size);
  for (let face = 0; face < 6; face++) {
    const axis = face >> 1;
    const sign = face & 1 ? -1 : 1;
    const [ua, va] = LOOKUP_AXES[face]!;
    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        const d = [0, 0, 0];
        d[axis] = sign;
        d[ua] = ((col + 0.5) / size) * 2 - 1;
        d[va] = ((row + 0.5) / size) * 2 - 1;
        const length = Math.hypot(d[0]!, d[1]!, d[2]!);
        const [x, y, z] = [d[0]! / length, d[1]! / length, d[2]! / length];
        let best = 0;
        let bestDistance = Infinity;
        for (let reach = 1; bestDistance === Infinity; reach++) {
          const [bx, by, bz] = [bin(x), bin(y), bin(z)];
          for (let i = bx - reach; i <= bx + reach; i++)
            for (let j = by - reach; j <= by + reach; j++)
              for (let k = bz - reach; k <= bz + reach; k++) {
                for (const v of buckets.get((i * bins + j) * bins + k) ?? []) {
                  const dd =
                    (positions[v * 3]! - x) ** 2 +
                    (positions[v * 3 + 1]! - y) ** 2 +
                    (positions[v * 3 + 2]! - z) ** 2;
                  if (dd < bestDistance) {
                    bestDistance = dd;
                    best = v;
                  }
                }
              }
        }
        lookup[face * size * size + row * size + col] = best;
      }
    }
  }
  return lookup;
}
