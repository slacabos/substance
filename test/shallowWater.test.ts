import { describe, expect, it } from 'vitest';
import { OCEAN_DEPTH } from '../src/core/oceanFFT';
import { oceanMesh } from '../src/core/oceanMesh';
import { oceanDetail } from '../src/core/oceanTuning';
import {
  SURFACE,
  createSurfaceState,
  stepSurface,
  surfaceVolume,
  waterVolume,
  vertexAreas,
  type SurfaceField,
} from '../src/core/shallowWater';
import { buildWaveKernel } from '../src/core/waveKernel';
import { createDeformGeometry } from '../src/materials/deform';

const mesh = oceanMesh(createDeformGeometry(true, oceanDetail(true)));
const count = mesh.positions.length / 3;
const dt = 1 / 60;

function field(depth: number, dry?: (i: number) => boolean): SurfaceField {
  const depths = new Float32Array(count).fill(depth);
  const wet = new Uint8Array(count).fill(1);
  if (dry) for (let i = 0; i < count; i++) if (dry(i)) wet[i] = 0;
  return {
    depth: depths,
    wet,
    kernel: buildWaveKernel(mesh.positions, mesh.neighbors, depths),
    areas: vertexAreas(mesh),
  };
}

/** Angle between vertex i and a unit direction. */
const angleTo = (i: number, [x, y, z]: number[]) =>
  Math.acos(
    Math.min(
      1,
      mesh.positions[i * 3]! * x! + mesh.positions[i * 3 + 1]! * y! + mesh.positions[i * 3 + 2]! * z!,
    ),
  );

/** A hand-of-god push centred on `at`: a broad Gaussian, as height acceleration. */
function push(at: number[], strength: number, radius = 0.14): Float32Array {
  const accel = new Float32Array(count);
  for (let i = 0; i < count; i++)
    accel[i] = strength * Math.exp(-(angleTo(i, at) ** 2) / (2 * radius * radius));
  return accel;
}

const deep = field(OCEAN_DEPTH);
const north = [0, 0, 1];

describe('interactive sea surface', () => {
  it('preserves vertical velocity across changing substep durations, apart from physical damping', () => {
    const state = createSurfaceState(count);
    let velocity = 0.01;
    state.hPrev.fill(-velocity * 0.024);
    for (const stepDt of [0.024, 0.013, 0.013, 0.024, 1 / 120, 0.024]) {
      stepSurface(mesh, deep, state, 0);
      stepSurface(mesh, deep, state, stepDt);
      const damping = (SURFACE.damping * stepDt) / 2;
      velocity *= (1 - damping) / (1 + damping);
      for (let i = 0; i < count; i++)
        expect((state.h[i]! - state.hPrev[i]!) / stepDt).toBeCloseTo(velocity, 6);
    }
  });

  it('stays flat when left alone and bounded under rough forcing', () => {
    const state = createSurfaceState(count);
    for (let i = 0; i < 60; i++) stepSurface(mesh, deep, state, dt);
    expect(state.h.every((v) => v === 0)).toBe(true);
    let seed = 7;
    const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
    const accel = new Float32Array(count);
    // grid-scale roughness: each height's difference from its neighbours' mean
    const energy = () => {
      let sum = 0;
      for (let i = 0; i < count; i++) {
        let mean = 0;
        for (let k = 0; k < 6; k++) mean += state.h[mesh.neighbors[i * 6 + k]!]! / 6;
        sum += (state.h[i]! - mean) ** 2;
      }
      return sum;
    };
    let afterForcing = 0;
    for (let step = 0; step < 900; step++) {
      if (step < 120) for (let i = 0; i < count; i++) accel[i] = random() * 0.1;
      stepSurface(mesh, deep, state, dt, step < 120 ? accel : undefined);
      if (step === 120) afterForcing = energy();
    }
    expect(afterForcing).toBeGreaterThan(0);
    expect(Number.isFinite(energy())).toBe(true);
    // grid-scale chatter dies out fast; what's left are slow long waves
    expect(energy()).toBeLessThan(afterForcing * 0.05);
  });

  it('sends a tap out as a ring at about the long-wave speed sqrt(gH)', () => {
    const state = createSurfaceState(count);
    // one-step impulse, about what a tap does: the centre starts down at 0.012 radii per second
    stepSurface(mesh, deep, state, dt, push(north, -0.012 / dt));
    const ringAt = (seconds: number) => {
      while (state.hPrev.length && (ringAt.t ?? 0) < seconds) {
        stepSurface(mesh, deep, state, dt);
        ringAt.t = (ringAt.t ?? 0) + dt;
      }
      // the leading edge: the furthest angle where the water has clearly moved
      let peak = 0;
      for (let i = 0; i < count; i++) peak = Math.max(peak, Math.abs(state.h[i]!));
      let front = 0;
      for (let i = 0; i < count; i++)
        if (Math.abs(state.h[i]!) > peak * 0.25) front = Math.max(front, angleTo(i, north));
      return front;
    };
    ringAt.t = 0 as number | undefined;
    const early = ringAt(1);
    const late = ringAt(3);
    const speed = (late - early) / 2;
    const longWave = Math.sqrt(SURFACE.gravity * OCEAN_DEPTH);
    expect(speed).toBeGreaterThan(longWave * 0.6);
    expect(speed).toBeLessThan(longWave * 1.3);
  });

  it('conserves water in the shallows and keeps it off an island', () => {
    const island = (i: number) => angleTo(i, [1, 0, 0]) < 0.25;
    const shallows = field(0.004, island);
    const state = createSurfaceState(count);
    for (let i = 0; i < count; i++) {
      if (island(i)) continue;
      const d = angleTo(i, [0.8, 0.6, 0]);
      state.h[i] = 0.003 * Math.exp(-(d * d) / 0.02);
    }
    state.hPrev.set(state.h);
    const before = surfaceVolume(state, shallows.areas);
    for (let step = 0; step < 600; step++) stepSurface(mesh, shallows, state, dt);
    const after = surfaceVolume(state, shallows.areas);
    expect(Math.abs(after - before)).toBeLessThan(Math.abs(before) * 1e-4);
    for (let i = 0; i < count; i++) if (island(i)) expect(state.h[i]).toBe(0);
    // the wave moved: it isn't still the starting bump
    expect(Math.max(...state.h)).toBeLessThan(0.0029);
  });

  it('fills a basin dug below the sea once a channel reaches it, and keeps every drop', () => {
    // a shallow sea, land around [1, 0, 0] standing 1 cm above it, and inside it a dry basin dug 3 mm below
    const land = [1, 0, 0];
    const basin = (i: number) => angleTo(i, land) < 0.15;
    const beach = field(0.004);
    const state = createSurfaceState(count);
    // ground at height `bed`, dry: its surface is the ground itself
    const ground = (i: number, bed: number) => {
      beach.depth[i] = -bed;
      state.h[i] = bed;
    };
    for (let i = 0; i < count; i++) {
      if (basin(i)) ground(i, -0.003);
      else if (angleTo(i, land) < 0.4) ground(i, 0.01);
    }
    state.hPrev.set(state.h);
    const column = (i: number) => beach.depth[i]! + state.h[i]!;
    const held = () => Math.max(...Array.from({ length: count }, (_, i) => (basin(i) ? column(i) : 0)));
    const before = waterVolume(beach, state);
    for (let step = 0; step < 120; step++) stepSurface(mesh, beach, state, dt);
    // the ridge holds the sea back
    expect(held()).toBeLessThan(SURFACE.dryColumn * 2);
    // dig a channel from the basin out to the sea, along the equator toward +y
    for (let i = 0; i < count; i++) {
      const [x, y, z] = [mesh.positions[i * 3]!, mesh.positions[i * 3 + 1]!, mesh.positions[i * 3 + 2]!];
      if (Math.abs(z) < 0.04 && y > 0 && x > 0.85 && !basin(i) && beach.depth[i]! < 0) {
        ground(i, -0.003);
        state.hPrev[i] = -0.003;
      }
    }
    for (let step = 0; step < 900; step++) stepSurface(mesh, beach, state, dt);
    // water ran in, and the sea gave up what the basin took
    expect(held()).toBeGreaterThan(0.001);
    const after = waterVolume(beach, state);
    expect(Math.abs(after - before)).toBeLessThan(before * 2e-3);
    // land above the water stayed dry
    for (let i = 0; i < count; i++)
      if (angleTo(i, land) < 0.4 && angleTo(i, land) > 0.2 && beach.depth[i]! < 0)
        expect(column(i)).toBeLessThan(SURFACE.dryColumn * 5);
  });

  it('builds a bigger wave from pushes that land before the water recovers', () => {
    const state = (pushes: number) => {
      const sea = createSurfaceState(count);
      const kick = push(north, -0.006 / dt);
      for (let step = 0; step < 240; step++)
        stepSurface(mesh, deep, sea, dt, step % 40 === 0 && step < pushes * 40 ? kick : undefined);
      return Math.max(...sea.h.map(Math.abs));
    };
    expect(state(3)).toBeGreaterThan(state(1) * 2);
  });
});
