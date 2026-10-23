import { describe, expect, it } from 'vitest';
import { approach, Spring, stepSprings, substeps } from '../src/core/spring';

describe('Spring', () => {
  it('settles on its target', () => {
    const s = new Spring(110, 7, 0, -0.2);
    stepSprings([s], 3);
    expect(s.value).toBeCloseTo(-0.2, 4);
    expect(s.isSettled()).toBe(true);
  });

  it('overshoots when underdamped, which is what makes slime jiggle', () => {
    const s = new Spring(110, 7, -0.25, 0);
    let max = -Infinity;
    for (let i = 0; i < 200; i++) {
      stepSprings([s], 1 / 120);
      max = Math.max(max, s.value);
    }
    expect(max).toBeGreaterThan(0.02);
  });

  it('stays stable with a long frame thanks to substeps', () => {
    const s = new Spring(240, 13, 0.8, 1);
    stepSprings([s], 0.05);
    expect(Number.isFinite(s.value)).toBe(true);
    expect(Math.abs(s.value)).toBeLessThan(2);
  });
});

describe('substeps', () => {
  it('never exceeds the max step and covers the whole frame', () => {
    const { count, h } = substeps(0.05, 0.006);
    expect(h).toBeLessThanOrEqual(0.006);
    expect(count * h).toBeCloseTo(0.05, 10);
  });

  it('always takes at least one step', () => {
    expect(substeps(0, 0.01).count).toBe(1);
  });
});

describe('approach', () => {
  it('is frame-rate independent', () => {
    const oneBig = approach(3, 0.1);
    const twoSmall = 1 - (1 - approach(3, 0.05)) ** 2;
    expect(oneBig).toBeCloseTo(twoSmall, 10);
  });
});
