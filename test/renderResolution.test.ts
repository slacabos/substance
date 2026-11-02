import { describe, expect, it } from 'vitest';
import { renderPixelRatio } from '../src/core/renderResolution';

describe('plane phone resolution', () => {
  it.each([
    [390, 844],
    [844, 390],
    [1024, 1366],
    [2560, 1440],
  ])('caps density and total pixels at %s by %s', (width, height) => {
    const ratio = renderPixelRatio(width, height, 3, false, true, true);
    expect(ratio).toBeLessThanOrEqual(1.5);
    expect(width * height * ratio ** 2).toBeLessThanOrEqual(1_500_000.000001);
  });
  it('does not upscale a low-density phone and restores orb quality', () => {
    expect(renderPixelRatio(390, 844, 1, false, true, true)).toBe(1);
    expect(renderPixelRatio(390, 844, 3, false, true, false)).toBe(2);
  });
  it('preserves the desktop supersampling policy in plane mode', () => {
    expect(renderPixelRatio(1280, 720, 1, true, false, true)).toBe(2);
  });
});
