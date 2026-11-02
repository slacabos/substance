import { describe, expect, it } from 'vitest';
import { planeUniform, ShapeMorph } from '../src/scene/shape';

describe('ShapeMorph', () => {
  it('leaves the plane exactly, so nothing drawn only for the plane stays on the orb', () => {
    const morph = new ShapeMorph();
    morph.set('plane', true);
    morph.update(1 / 60);
    expect(planeUniform.value).toBe(1);
    for (const shape of ['orb', 'cube'] as const) {
      morph.set('plane', true);
      morph.set(shape, false);
      for (let i = 0; i < 60 * 4; i++) morph.update(1 / 60);
      expect(planeUniform.value).toBe(0);
      expect(morph.plane).toBe(0);
    }
  });

  it('reaches the plane exactly', () => {
    const morph = new ShapeMorph();
    morph.set('plane', false);
    for (let i = 0; i < 60 * 4; i++) morph.update(1 / 60);
    expect(planeUniform.value).toBe(1);
  });
});
