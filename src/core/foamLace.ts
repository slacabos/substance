// Churned foam is white water full of round holes, the bubbles, of many sizes. This bakes them once, as a tiling
// texture the ocean shader projects onto the orb (nodes/ocean.ts). Each texel holds its distance to the nearest
// bubble's centre (F1 of 2D Worley noise) and that bubble's random number, so the shader can open each bubble to
// its own radius: small holes in thick foam, growing and merging into a thin web as it thins.

/** Distances are stored divided by this, so they fit 0 to 1. */
export const LACE_RANGE = 1.2;

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
 * At (u, v), in cell widths: the distance to the nearest bubble's centre, and that bubble's random number. The
 * pattern repeats every `cells`.
 */
export function laceAt(u: number, v: number, cells: number, seed = 1): { distance: number; cell: number } {
  const cx = Math.floor(u);
  const cy = Math.floor(v);
  let nearest = Infinity;
  let cell = 0;
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) {
      const [px, py] = cellHash(cx + dx, cy + dy, cells, seed, 0);
      const d = (cx + dx + px - u) ** 2 + (cy + dy + py - v) ** 2;
      if (d < nearest) {
        nearest = d;
        cell = cellHash(cx + dx, cy + dy, cells, seed, 1)[0];
      }
    }
  return { distance: Math.sqrt(nearest), cell };
}

/** A size x size tile, `cells` cells across, two bytes a texel: distance / LACE_RANGE, and the bubble's number. */
export function bakeFoamLace(size: number, cells: number, seed = 1): Uint8Array {
  const data = new Uint8Array(size * size * 2);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const { distance, cell } = laceAt(((x + 0.5) / size) * cells, ((y + 0.5) / size) * cells, cells, seed);
      data[(y * size + x) * 2] = Math.round(Math.min(1, distance / LACE_RANGE) * 255);
      data[(y * size + x) * 2 + 1] = Math.round(cell * 255);
    }
  return data;
}
