import type { WebGPURenderer } from 'three/webgpu';
import { describe, expect, it } from 'vitest';
import { oceanMesh } from '../src/core/oceanMesh';
import {
  createSandState,
  digSand,
  ploughSand,
  relaxSand,
  reposeSlope,
  SAND,
  sandMesh,
  sandVolume,
  smoothSand,
  type SandMesh,
  type SandState,
} from '../src/core/sand';
import { createDeformGeometry } from '../src/materials/deform';
import { SandField } from '../src/materials/sandField';
import { SandGrains } from '../src/materials/sandGrains';

const mesh = sandMesh(oceanMesh(createDeformGeometry(true, 48)));
const count = mesh.areas.length;
const north: [number, number, number] = [0, 1, 0];

/** The steepest slope between any two neighbours. */
function steepest(m: SandMesh, s: SandState): number {
  let most = 0;
  for (let i = 0; i < count; i++)
    for (let k = 0; k < 6; k++) {
      const j = m.neighbors[i * 6 + k]!;
      if (j === i) continue;
      const p = m.positions;
      const length = Math.hypot(
        p[i * 3]! - p[j * 3]!,
        p[i * 3 + 1]! - p[j * 3 + 1]!,
        p[i * 3 + 2]! - p[j * 3 + 2]!,
      );
      most = Math.max(most, Math.abs(s.bed[i]! - s.bed[j]!) / length);
    }
  return most;
}

/** A sharp cone of sand at the north pole, slope 1.5: far steeper than any sand can stand. */
function cone(wet: number): SandState {
  const state = createSandState(count);
  for (let i = 0; i < count; i++) {
    const angle = Math.acos(Math.min(1, mesh.positions[i * 3 + 1]!));
    state.bed[i] = Math.max(0, 0.12 - angle * 1.5);
  }
  state.wet.fill(wet);
  return state;
}

describe('sand', () => {
  it('slides down to its angle of repose and no further, without losing any', () => {
    const state = cone(0);
    const before = sandVolume(mesh, state);
    for (let i = 0; i < 400; i++) relaxSand(mesh, state);
    expect(Math.abs(sandVolume(mesh, state) - before)).toBeLessThan(1e-6 * Math.abs(before) + 1e-12);
    expect(steepest(mesh, state)).toBeLessThan(reposeSlope(0) * 1.08);
    // it didn't flatten: it stopped at the repose slope
    expect(steepest(mesh, state)).toBeGreaterThan(reposeSlope(0) * 0.8);
  });

  it('holds steeper slopes damp than dry, and slumps under water', () => {
    expect(reposeSlope(0.5)).toBeGreaterThan(reposeSlope(0));
    expect(reposeSlope(1)).toBeLessThan(reposeSlope(0));
    const dry = cone(0);
    const damp = cone(0.5);
    for (let i = 0; i < 400; i++) {
      relaxSand(mesh, dry);
      relaxSand(mesh, damp);
    }
    expect(steepest(mesh, damp)).toBeGreaterThan(steepest(mesh, dry) * 1.2);
    // and damp sand keeps a taller pile
    const peak = (s: SandState) => Math.max(...s.bed);
    expect(peak(damp)).toBeGreaterThan(peak(dry));
  });

  it('digs a crater with a rim and ploughs a furrow with a ridge ahead, making no sand', () => {
    const state = createSandState(count);
    digSand(mesh, state, north, 0.02, 60);
    expect(Math.abs(sandVolume(mesh, state))).toBeLessThan(1e-9);
    const at = (dir: [number, number, number]) => {
      let best = 0;
      let index = 0;
      for (let i = 0; i < count; i++) {
        const dot =
          mesh.positions[i * 3]! * dir[0] +
          mesh.positions[i * 3 + 1]! * dir[1] +
          mesh.positions[i * 3 + 2]! * dir[2];
        if (dot > best) [best, index] = [dot, i];
      }
      return state.bed[index]!;
    };
    expect(at(north)).toBeLessThan(-0.015);
    // the rim, about where the footprint turns over (x = 2: 1 - cos a = 2 / 60)
    const a = Math.acos(1 - 2 / 60);
    expect(at([Math.sin(a), Math.cos(a), 0])).toBeGreaterThan(0);

    const ploughed = createSandState(count);
    ploughSand(mesh, ploughed, north, [1, 0, 0], 0.01, 60);
    expect(Math.abs(sandVolume(mesh, ploughed))).toBeLessThan(1e-9);
    // a trench along the track (x) with banks either side (z), and a little more sand ahead than behind
    const near = (x: number, z: number) => {
      let best = -1;
      let index = 0;
      const length = Math.hypot(x, 1, z);
      for (let i = 0; i < count; i++) {
        const dot =
          (mesh.positions[i * 3]! * x + mesh.positions[i * 3 + 1]! + mesh.positions[i * 3 + 2]! * z) / length;
        if (dot > best) [best, index] = [dot, i];
      }
      return ploughed.bed[index]!;
    };
    const width = 1 / Math.sqrt(2 * 60);
    expect(near(0, 0)).toBeLessThan(-0.005);
    expect(near(0, 1.8 * width)).toBeGreaterThan(0);
    expect(near(0, -1.8 * width)).toBeGreaterThan(0);
    expect(near(width, 0)).toBeGreaterThan(near(-width, 0));
  });

  it('is smoothed only where water runs over it', () => {
    const state = createSandState(count);
    let seed = 5;
    const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.002;
    for (let i = 0; i < count; i++) state.bed[i] = random();
    const flow = new Float32Array(count);
    for (let i = 0; i < count; i++) flow[i] = mesh.positions[i * 3 + 1]! > 0 ? 1 : 0;
    const roughness = (half: (y: number) => boolean) => {
      let sum = 0;
      for (let i = 0; i < count; i++) {
        if (!half(mesh.positions[i * 3 + 1]!)) continue;
        for (let k = 0; k < 6; k++) sum += Math.abs(state.bed[i]! - state.bed[mesh.neighbors[i * 6 + k]!]!);
      }
      return sum;
    };
    const north0 = roughness((y) => y > 0.1);
    const south0 = roughness((y) => y < -0.1);
    const before = sandVolume(mesh, state);
    for (let i = 0; i < 20; i++) smoothSand(mesh, state, flow, 0.1);
    expect(roughness((y) => y > 0.1)).toBeLessThan(north0 * 0.5);
    expect(roughness((y) => y < -0.1)).toBeCloseTo(south0, 10);
    // to float precision
    expect(Math.abs(sandVolume(mesh, state) - before)).toBeLessThan(1e-10);
    expect(SAND.layer).toBeGreaterThan(0.05);
  });

  it('runs on the GPU in one dispatch a frame: landings, the hand, relaxation and slopes', () => {
    const calls: unknown[][] = [];
    const renderer = {
      compute: (nodes: unknown) => calls.push(Array.isArray(nodes) ? nodes : [nodes]),
    } as unknown as WebGPURenderer;
    const field = new SandField(renderer, createDeformGeometry(true, 12));
    const grains = new SandGrains(renderer, true, field);
    field.update(1 / 60);
    // landings, two relaxation passes and the slope
    expect(calls.at(-1)).toHaveLength(4);
    field.add({ dir: north, amount: 0.01, sharpness: 100 });
    field.update(1 / 60);
    expect(calls.at(-1)).toHaveLength(5);
    grains.throw(north, [1, 0, 0], 1, 0.5);
    grains.update(1 / 60);
    // the throw, then the flight
    expect(calls.at(-1)).toHaveLength(2);
    grains.update(1 / 60);
    expect(calls.at(-1)).toHaveLength(1);
    field.update(0);
    expect(calls).toHaveLength(4);
  });
});
