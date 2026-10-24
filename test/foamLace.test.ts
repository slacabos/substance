import { describe, expect, it } from 'vitest';
import { bakeFoamLace, laceAt } from '../src/core/foamLace';

describe('foam lace', () => {
  it('repeats seamlessly across the tile', () => {
    for (const [u, v] of [
      [0.1, 0.2],
      [3.7, 5.9],
      [7.95, 0.05],
    ] as const) {
      const here = laceAt(u, v, 8);
      for (const there of [laceAt(u + 8, v, 8), laceAt(u, v - 8, 8)]) {
        expect(there.edge).toBeCloseTo(here.edge, 10);
        expect(there.cell).toBe(here.cell);
      }
    }
  });

  it('is a web: thin edges at zero around wider cells', () => {
    const data = bakeFoamLace(128, 8).filter((_, i) => i % 2 === 0);
    const sorted = [...data].sort((a, b) => a - b);
    expect(sorted[0]).toBeLessThan(8);
    // edges are a small share of the tile, cell interiors most of it
    expect(sorted[Math.floor(data.length * 0.1)]).toBeGreaterThan(10);
    expect(sorted[data.length - 1]).toBeGreaterThan(120);
  });

  it('gives each cell one number of its own', () => {
    const cells = new Set<number>();
    for (let k = 0; k < 64; k++) cells.add(laceAt((k % 8) + 0.5, Math.floor(k / 8) + 0.5, 8).cell);
    // most of the 64 cells differ
    expect(cells.size).toBeGreaterThan(50);
    // and the number is constant across a cell's interior
    const { cell } = laceAt(2.5, 3.5, 8);
    expect(laceAt(2.52, 3.48, 8).cell).toBe(cell);
  });
});
