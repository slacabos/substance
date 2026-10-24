import { describe, expect, it } from 'vitest';
import { bakeFoamLace, laceAt } from '../src/core/foamLace';

describe('foam lace', () => {
  it('repeats seamlessly across the tile', () => {
    for (const [u, v] of [
      [0.1, 0.2],
      [3.7, 5.9],
      [7.95, 0.05],
    ] as const) {
      expect(laceAt(u + 8, v, 8)).toBeCloseTo(laceAt(u, v, 8), 10);
      expect(laceAt(u, v - 8, 8)).toBeCloseTo(laceAt(u, v, 8), 10);
    }
  });

  it('is a web: thin edges at zero around wider cells', () => {
    const data = bakeFoamLace(128, 8);
    const sorted = [...data].sort((a, b) => a - b);
    expect(sorted[0]).toBeLessThan(8);
    // edges are a small share of the tile, cell interiors most of it
    expect(sorted[Math.floor(data.length * 0.1)]).toBeGreaterThan(10);
    expect(sorted[data.length - 1]).toBeGreaterThan(120);
  });
});
