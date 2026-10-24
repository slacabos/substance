import { describe, expect, it } from 'vitest';
import { strandEstimate } from '../src/materials/fur';

describe('readout figures', () => {
  it('estimates strands from lattice density', () => {
    expect(strandEstimate(64)).toBe(36);
  });
});
