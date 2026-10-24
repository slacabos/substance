import { Group, Vector3 } from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import { describe, expect, it } from 'vitest';
import { FLUFF, fluffFrame } from '../src/core/fluff';
import { Fluff, strandCount } from '../src/materials/fluff';
import { FluffSim } from '../src/materials/fluffSim';

function stubRenderer() {
  const calls: unknown[] = [];
  const renderer = { compute: (node: unknown) => calls.push(node) } as unknown as WebGPURenderer;
  return { renderer, calls };
}

describe('fluff simulation', () => {
  it('keeps existing velocity when frame timing changes and caps stalled frames', () => {
    const velocity = 0.4;
    let previous = 1 / FLUFF.physics.rate;
    let displacement = velocity * previous;
    for (const dt of [1 / 60, 1 / 144, 0.024, 0.026, 1 / 30]) {
      const frame = fluffFrame(dt, previous)!;
      displacement *= frame.historyScale;
      expect(displacement / frame.step).toBeCloseTo(velocity, 12);
      expect(frame.step).toBeLessThanOrEqual(1 / FLUFF.physics.rate);
      previous = frame.step;
    }
    expect(fluffFrame(2, previous)!.steps).toBe(FLUFF.physics.maxSteps);
    for (const dt of [0, -1, NaN, Infinity]) expect(fluffFrame(dt, previous)).toBeNull();
  });

  it('rebases the strand history once per update and clears contact on reset', () => {
    const { renderer, calls } = stubRenderer();
    const roots = new Float32Array([0, 1, 0, 0, 0, 1]);
    const rand = new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    const anchor = new Group();
    const sim = new FluffSim(renderer, roots, rand, FLUFF.points(true));
    sim.reset(anchor);
    sim.update(anchor, 0, 1 / 144);
    expect(sim.uniforms.historyScale.value).toBeCloseTo(1 / 288 / (1 / 240));
    sim.update(anchor, 1, 0);
    expect(calls).toHaveLength(2);
    sim.update(anchor, 1, 1 / 60);
    expect(sim.uniforms.historyScale.value).toBeCloseTo(1 / 240 / (1 / 288));
    sim.uniforms.brushAmount.value = 1;
    sim.uniforms.press.value = 1;
    sim.reset(anchor);
    expect(sim.uniforms.historyScale.value).toBe(1);
    expect(sim.uniforms.brushAmount.value).toBe(0);
    expect(sim.uniforms.press.value).toBe(0);
  });
});

describe('fluff material', () => {
  it.each([false, true])('retains the original individual-strand coat (touch: %s)', (coarse) => {
    const { renderer, calls } = stubRenderer();
    const anchor = new Group();
    const fluff = new Fluff(coarse, renderer, anchor);
    const count = strandCount(FLUFF.density);
    expect(fluff.worldObjects[0]!.geometry.instanceCount).toBe(count);
    expect(fluff.readout()).toEqual(['Fluff', `≈ ${Math.round(count / 1000)}k strands`, 'keratin n ≈ 1.55']);
    fluff.down(new Vector3(0, 0, 1));
    fluff.update({ anchor, dt: 1 / 60, time: 0 });
    expect(fluff['sim'].uniforms.press.value).toBeGreaterThan(0);
    fluff.enter();
    fluff.update({ anchor, dt: 1 / 60, time: 1 });
    expect(fluff['sim'].uniforms.press.value).toBe(0);
    fluff.worldObjects[0]!.visible = false;
    const before = calls.length;
    fluff.update({ anchor, dt: 1 / 60, time: 2 });
    expect(calls).toHaveLength(before);
  });
});
