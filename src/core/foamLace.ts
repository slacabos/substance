// Churned foam is a lace of thin films around bubbles. This bakes that web once, as a tiling texture the
// ocean shader projects onto the orb (nodes/ocean.ts). Each texel holds its distance to the nearest edge between
// jittered cells (F2 - F1 of 2D Worley noise), so the shader can thicken or thin the web by where it cuts, and
// a random number for the cell it's in, so bubbles open by different amounts.

/** Edge distances are stored divided by this, so they fit 0 to 1. */
export const LACE_RANGE = 0.8;

/** Hash of a cell (wrapped to the tile) to two numbers in 0 to 1, distinct per `salt`. */
function cellHash(cx: number, cy: number, cells: number, seed: number, salt: number): [number, number] {
  const x = ((cx % cells) + cells) % cells;
  const y = ((cy % cells) + cells) % cells;
  let h = Math.imul(x * 73856093 + (seed * 2 + salt) * 83492791, 1) ^ Math.imul(y, 19349663);
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
  h ^= h >>> 16;
  return [(h & 0xffff) / 0x10000, (h >>> 16) / 0x10000];
}

/**
 * At (u, v), in cell widths: the distance to the nearest edge between cells, and the random number of the cell
 * it's in. The pattern repeats every `cells`.
 */
export function laceAt(u: number, v: number, cells: number, seed = 1): { edge: number; cell: number } {
  const cx = Math.floor(u);
  const cy = Math.floor(v);
  let f1 = Infinity;
  let f2 = Infinity;
  let cell = 0;
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) {
      const [px, py] = cellHash(cx + dx, cy + dy, cells, seed, 0);
      const d = (cx + dx + px - u) ** 2 + (cy + dy + py - v) ** 2;
      f2 = Math.min(f2, Math.max(f1, d));
      if (d < f1) cell = cellHash(cx + dx, cy + dy, cells, seed, 1)[0];
      f1 = Math.min(f1, d);
    }
  return { edge: Math.sqrt(f2) - Math.sqrt(f1), cell };
}

/** A size x size tile, `cells` cells across, two bytes a texel: edge distance / LACE_RANGE, and the cell's number. */
export function bakeFoamLace(size: number, cells: number, seed = 1): Uint8Array {
  const data = new Uint8Array(size * size * 2);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const { edge, cell } = laceAt(((x + 0.5) / size) * cells, ((y + 0.5) / size) * cells, cells, seed);
      data[(y * size + x) * 2] = Math.round(Math.min(1, edge / LACE_RANGE) * 255);
      data[(y * size + x) * 2 + 1] = Math.round(cell * 255);
    }
  return data;
}
