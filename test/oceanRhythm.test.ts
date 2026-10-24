import { describe, expect, it } from 'vitest';
import { TapRhythm } from '../src/core/oceanRhythm';

describe('tap rhythm', () => {
  it('pushes harder with each tap that keeps the beat, up to a limit', () => {
    const rhythm = new TapRhythm();
    const boosts = [0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2].map((t) => rhythm.tap(t));
    // the first two taps set the beat
    expect(boosts.slice(0, 3)).toEqual([1, 1, 1 + TapRhythm.GAIN]);
    for (let i = 3; i < boosts.length; i++) expect(boosts[i]).toBeGreaterThanOrEqual(boosts[i - 1]!);
    expect(boosts.at(-1)).toBeCloseTo(1 + TapRhythm.GAIN * TapRhythm.MAX_STREAK);
  });

  it('follows a drifting tempo but starts over off the beat or after a pause', () => {
    const rhythm = new TapRhythm();
    for (const t of [0, 0.8, 1.6, 2.35, 3.05]) rhythm.tap(t);
    expect(rhythm.tap(3.7)).toBeGreaterThan(1.5);
    // far too soon
    expect(rhythm.tap(3.8)).toBe(1);
    rhythm.reset();
    for (const t of [10, 10.5, 11]) rhythm.tap(t);
    // a long pause
    expect(rhythm.tap(14)).toBe(1);
  });
});
