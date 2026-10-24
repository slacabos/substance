import { IcosahedronGeometry, Vector3 } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import type { WebGPURenderer } from 'three/webgpu';
import { describe, expect, it } from 'vitest';
import {
  createFluidState,
  FLUID,
  fluidGrid,
  fromFixed,
  particleVolume,
  splineWeights,
  stepFluid,
  toFixed,
} from '../src/core/oceanFluid';
import { OCEAN_DEFAULTS } from '../src/core/oceanTuning';
import { createDeformUniforms } from '../src/materials/deform';
import { OceanFFT } from '../src/materials/oceanFFT';
import { OceanFluid } from '../src/materials/oceanFluid';
import { OceanSim } from '../src/materials/oceanSim';

/** A ball of particles two to a cell side, centred at c. */
function blob(c: [number, number, number], radius: number, spacing: number) {
  const points: number[] = [];
  for (let x = -radius; x <= radius; x += spacing)
    for (let y = -radius; y <= radius; y += spacing)
      for (let z = -radius; z <= radius; z += spacing)
        if (x * x + y * y + z * z <= radius * radius) points.push(c[0] + x, c[1] + y, c[2] + z);
  const state = createFluidState(points.length / 3);
  state.x.set(points);
  return state;
}

const mean = (a: Float64Array, d: number) => {
  let sum = 0;
  for (let i = d; i < a.length; i += 3) sum += a[i]!;
  return sum / (a.length / 3);
};

describe('ocean fluid (MLS-MPM)', () => {
  it('uses B-spline weights that sum to one and reproduce linear fields', () => {
    for (const f of [0.5, 0.73, 1, 1.21, 1.4999]) {
      const w = splineWeights(f);
      expect(w[0] + w[1] + w[2]).toBeCloseTo(1, 12);
      // the weighted mean of the cells is the particle
      expect(w[0] * 0 + w[1] * 1 + w[2] * 2).toBeCloseTo(f, 12);
      for (const x of w) expect(x).toBeGreaterThanOrEqual(0);
    }
  });

  it('round-trips the grid sums through fixed point', () => {
    for (const x of [0, 1e-5, -0.37, 3.2]) expect(Math.abs(fromFixed(toFixed(x)) - x)).toBeLessThan(1e-6);
  });

  it('keeps momentum without gravity, even while pressure pushes particles apart', () => {
    const grid = fluidGrid(0.1);
    const state = blob([0.3, 0.2, -0.1], 0.15, grid.dx / 2);
    state.J.fill(0.85);
    for (let p = 0; p < state.J.length; p++) state.v.set([0.2, -0.1, 0.05], p * 3);
    const before = [0, 1, 2].map((d) => mean(state.v, d));
    for (let i = 0; i < 10; i++) stepFluid(state, grid, FLUID.maxStep, 0);
    const after = [0, 1, 2].map((d) => mean(state.v, d));
    for (let d = 0; d < 3; d++) expect(after[d]).toBeCloseTo(before[d]!, 6);
  });

  it('pushes compressed water apart and leaves water at rest still', () => {
    const grid = fluidGrid(0.1);
    const spread = (s: ReturnType<typeof blob>) => {
      let sum = 0;
      for (let p = 0; p < s.J.length; p++) sum += Math.hypot(s.x[p * 3]!, s.x[p * 3 + 1]!, s.x[p * 3 + 2]!);
      return sum / s.J.length;
    };
    const squeezed = blob([0, 0, 0], 0.15, grid.dx / 2);
    squeezed.J.fill(0.8);
    const start = spread(squeezed);
    for (let i = 0; i < 20; i++) stepFluid(squeezed, grid, FLUID.maxStep, 0);
    expect(spread(squeezed)).toBeGreaterThan(start + 1e-4);

    const still = blob([0, 0, 0], 0.15, grid.dx / 2);
    for (let i = 0; i < 20; i++) stepFluid(still, grid, FLUID.maxStep, 0);
    for (const v of still.v) expect(Math.abs(v)).toBeLessThan(1e-9);
  });

  it('falls toward the orb under radial gravity', () => {
    const grid = fluidGrid(0.1);
    // well inside the box, whose outer two cells hold still
    const state = blob([0, 1, 0], 0.06, grid.dx / 2);
    const steps = 60;
    for (let i = 0; i < steps; i++) stepFluid(state, grid, FLUID.maxStep);
    const t = steps * FLUID.maxStep;
    expect(mean(state.v, 1)).toBeCloseTo(-FLUID.gravity * t, 2);
    // and straight down: radial gravity only converges a little toward the axis
    expect(Math.abs(mean(state.v, 0))).toBeLessThan(0.03 * FLUID.gravity * t);
    expect(particleVolume(grid)).toBeCloseTo((grid.dx / 2) ** 3, 12);
  });

  it('runs on the GPU alone: starts, substeps by the frame time, then landings, in one dispatch', () => {
    const calls: unknown[][] = [];
    const renderer = {
      compute: (nodes: unknown) => calls.push(Array.isArray(nodes) ? nodes : [nodes]),
    } as unknown as WebGPURenderer;
    const source = new IcosahedronGeometry(1, 6);
    source.deleteAttribute('uv');
    const sim = new OceanSim(renderer, mergeVertices(source));
    const fluid = new OceanFluid(renderer, true, {
      buffers: sim.renderBuffers(),
      landings: sim.landings,
      field: sim.field,
      waves: new OceanFFT(renderer, true, OCEAN_DEFAULTS),
      uniforms: createDeformUniforms(),
      coastal: false,
    });
    fluid.splash(new Vector3(0, 0, 1), 0.5, new Vector3(1, 0, 0));
    fluid.update(1 / 60);
    // emission and drag splashes, four substeps of four passes (clear, to grid, grid, to particles), landings
    expect(calls.at(-1)).toHaveLength(2 + 4 * 4 + 1);
    fluid.update(1 / 20);
    expect(calls.at(-1)).toHaveLength(2 + 6 * 4 + 1);
    fluid.update(0);
    expect(calls).toHaveLength(2);
  });
});
