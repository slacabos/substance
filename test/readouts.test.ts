import { describe, expect, it } from 'vitest';
import { beaufortFor } from '../src/materials/ocean';
import { strandEstimate } from '../src/materials/fur';

describe('readout figures', () => {
  it('maps a calm ocean to Beaufort 2 and a full storm to about 8', () => {
    expect(beaufortFor(1)).toBe(2);
    expect(beaufortFor(2.4)).toBe(8);
  });

  it('estimates strands from lattice density', () => {
    expect(strandEstimate(64)).toBe(36);
  });
});
