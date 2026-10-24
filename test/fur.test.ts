import { Group, Vector3 } from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import { describe, expect, it } from 'vitest';
import { createFurCoat, furFrame, furGroom, FUR } from '../src/core/fur';
import { Fur, strandCount } from '../src/materials/fur';
import { FurSim } from '../src/materials/furSim';

const coat = createFurCoat(1200, 128);

describe('plush coat generation', () => {
  it('keeps both layers and their pigment variation stable across resets', () => {
    const repeat = createFurCoat(coat.count, coat.guideCount);
    expect(repeat).toEqual(coat);
    expect(createFurCoat(coat.count, coat.guideCount, 42).roots).not.toEqual(coat.roots);
    let guards = 0;
    for (let i = 0; i < coat.count; i++) {
      const at = i * 4;
      expect(Math.hypot(...coat.roots.subarray(at, at + 3))).toBeCloseTo(1, 6);
      const guard = coat.traits[at + 3]!;
      guards += guard;
      const range = guard ? FUR.guard : FUR.undercoat;
      expect(coat.roots[at + 3]).toBeGreaterThanOrEqual(range.min - 1e-8);
      expect(coat.roots[at + 3]).toBeLessThanOrEqual(range.max + 1e-8);
    }
    expect(guards / coat.count).toBeCloseTo(1 - FUR.undercoat.share, 6);
  });

  it('blends the actual three nearest guides with nonnegative weights that sum to one', () => {
    for (let i = 0; i < coat.count; i += 13) {
      const at = i * 4;
      const distance = (j: number) =>
        (coat.roots[at]! - coat.guideRoots[j * 4]!) ** 2 +
        (coat.roots[at + 1]! - coat.guideRoots[j * 4 + 1]!) ** 2 +
        (coat.roots[at + 2]! - coat.guideRoots[j * 4 + 2]!) ** 2;
      const nearest = Array.from({ length: coat.guideCount }, (_, j) => j).sort(
        (a, b) => distance(a) - distance(b),
      );
      expect([...coat.guides.subarray(at, at + 3)]).toEqual(nearest.slice(0, 3));
      expect(coat.weights[at]! + coat.weights[at + 1]! + coat.weights[at + 2]!).toBeCloseTo(1, 6);
      for (let j = 0; j < 3; j++) expect(coat.weights[at + j]).toBeGreaterThan(0);
    }
  });

  it('breaks up ordered rows and periodic long-hair bands without leaving one side sparse', () => {
    let latitudeChange = 0;
    let periodicLayers = 0;
    const mean = [0, 0, 0];
    for (let i = 0; i < coat.count; i++) {
      for (let d = 0; d < 3; d++) mean[d]! += coat.roots[i * 4 + d]! / coat.count;
      if (i > 0) latitudeChange += Math.abs(coat.roots[i * 4 + 1]! - coat.roots[(i - 1) * 4 + 1]!);
      if (i >= 5 && coat.traits[i * 4 + 3] === coat.traits[(i - 5) * 4 + 3]) periodicLayers++;
    }
    expect(latitudeChange / (coat.count - 1)).toBeGreaterThan(0.3);
    expect(periodicLayers / (coat.count - 5)).toBeLessThan(0.8);
    for (const component of mean) expect(Math.abs(component)).toBeLessThan(0.05);
  });

  it('has finite, tangential and continuous lay through poles and rounded cube transitions', () => {
    const directions = [
      new Vector3(0, 1, 0),
      new Vector3(0, -1, 0),
      new Vector3(1, 0, 0),
      new Vector3(1, 1, 0).normalize(),
      new Vector3(1, 1, 1).normalize(),
    ];
    for (const p of directions) {
      const cubeN = p
        .clone()
        .set(
          Math.sign(p.x) * Math.abs(p.x) ** 9,
          Math.sign(p.y) * Math.abs(p.y) ** 9,
          Math.sign(p.z) * Math.abs(p.z) ** 9,
        )
        .normalize();
      for (const morph of [0, 0.25, 0.5, 0.75, 1]) {
        const n = p.clone().lerp(cubeN, morph).normalize();
        const lay = new Vector3(...furGroom(p.toArray(), n.toArray()));
        expect(lay.toArray().every(Number.isFinite)).toBe(true);
        expect(lay.dot(n)).toBeCloseTo(0, 10);
        const near = p
          .clone()
          .add(new Vector3(1e-6, 0, 1e-6))
          .normalize();
        const nearbyLay = new Vector3(...furGroom(near.toArray(), n.toArray()));
        expect(nearbyLay.distanceTo(lay)).toBeLessThan(1e-5);
      }
    }
  });
});

describe('fur timing', () => {
  it('uses equivalent integration steps at 30, 60 and 120 FPS and bounds a stalled frame', () => {
    for (const fps of [30, 60, 120]) {
      const frame = furFrame(1 / fps, 1 / FUR.physics.rate)!;
      expect(frame.step).toBeCloseTo(1 / FUR.physics.rate, 12);
      expect(frame.steps * fps).toBe(FUR.physics.rate);
      expect(frame.historyScale).toBeCloseTo(1, 12);
    }
    const stalled = furFrame(5, 1 / 240)!;
    expect(stalled.steps).toBe(FUR.physics.maxSteps);
    expect(stalled.step).toBeLessThanOrEqual(1 / FUR.physics.rate);
    for (const dt of [0, -0.1, NaN, Infinity]) expect(furFrame(dt, 1 / 240)).toBeNull();
  });

  it('preserves an existing velocity when the substep duration changes repeatedly', () => {
    let previousStep = 1 / 240;
    const velocity = 0.37;
    let displacement = velocity * previousStep;
    for (const dt of [1 / 144, 0.024, 0.026, 1 / 30, 1 / 120, 0.001]) {
      const frame = furFrame(dt, previousStep)!;
      displacement *= frame.historyScale;
      expect(displacement / frame.step).toBeCloseTo(velocity, 12);
      previousStep = frame.step;
    }
  });

  it('dispatches only valid frames and resets history and contact state on re-entry', () => {
    const calls: unknown[] = [];
    const renderer = { compute: (node: unknown) => calls.push(node) } as unknown as WebGPURenderer;
    const sim = new FurSim(renderer, coat.guideRoots, 8);
    const anchor = new Group();
    sim.reset(anchor);
    sim.update(anchor, 1 / 144);
    expect(sim.uniforms.historyScale.value).toBeCloseTo(1 / 288 / (1 / 240));
    for (const dt of [0, -1, NaN]) sim.update(anchor, dt);
    expect(calls).toHaveLength(2);
    sim.update(anchor, 1 / 60);
    expect(sim.uniforms.historyScale.value).toBeCloseTo(1 / 240 / (1 / 288));
    sim.uniforms.brushAmount.value = 1;
    sim.uniforms.press.value = 1;
    sim.reset(anchor);
    expect(sim.uniforms.historyScale.value).toBe(1);
    expect(sim.uniforms.brushAmount.value).toBe(0);
    expect(sim.uniforms.press.value).toBe(0);
  });
});

describe('fur material', () => {
  it.each([false, true])('reports rendered fibres rather than simulated guides (touch: %s)', (coarse) => {
    const renderer = { compute: () => {} } as unknown as WebGPURenderer;
    const fur = new Fur(coarse, renderer, new Group());
    const count = strandCount(coarse ? FUR.touchDensity : FUR.density);
    expect(fur.worldObjects[0]!.geometry.instanceCount).toBe(count);
    expect(fur.readout()[1]).toBe(`≈ ${Math.round(count / 1000)}k strands`);
    fur.down(new Vector3(0, 0, 1));
    fur.enter();
    fur.update({ anchor: new Group(), dt: 1 / 60, time: 0 });
    expect(fur['sim'].uniforms.press.value).toBe(0);
  });
});
