// Churned foam is a lace of thin films around bubbles. This bakes that web once, as a tiling texture the
// ocean shader projects onto the orb (nodes/ocean.ts): each texel holds its distance to the nearest edge between
// jittered cells (F2 - F1 of 2D Worley noise), so the shader can thicken or thin the web by where it cuts.

/** Edge distances are stored divided by this, so they fit 0 to 1. */
export const LACE_RANGE = 0.8;

/** A cell's jittered point, 0 to 1 within the cell; the same for every repeat of the tile. */
function point(cx: number, cy: number, cells: number, seed: number): [number, number] {
  const x = ((cx % cells) + cells) % cells;
  const y = ((cy % cells) + cells) % cells;
  let h = Math.imul(x * 73856093 + seed * 83492791, 1) ^ Math.imul(y, 19349663);
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
  h ^= h >>> 16;
  return [(h & 0xffff) / 0x10000, (h >>> 16) / 0x10000];
}

/** Distance from (u, v), in cell widths, to the nearest edge between cells. The pattern repeats every `cells`. */
export function laceAt(u: number, v: number, cells: number, seed = 1): number {
  const cx = Math.floor(u);
  const cy = Math.floor(v);
  let f1 = Infinity;
  let f2 = Infinity;
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) {
      const [px, py] = point(cx + dx, cy + dy, cells, seed);
      const d = (cx + dx + px - u) ** 2 + (cy + dy + py - v) ** 2;
      f2 = Math.min(f2, Math.max(f1, d));
      f1 = Math.min(f1, d);
    }
  return Math.sqrt(f2) - Math.sqrt(f1);
}

/** A size x size tile, `cells` cells across, as bytes (edge distance / LACE_RANGE). */
export function bakeFoamLace(size: number, cells: number, seed = 1): Uint8Array {
  const data = new Uint8Array(size * size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const d = laceAt(((x + 0.5) / size) * cells, ((y + 0.5) / size) * cells, cells, seed);
      data[y * size + x] = Math.round(Math.min(1, d / LACE_RANGE) * 255);
    }
  return data;
}
