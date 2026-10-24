// Splashes as a particle fluid: MLS-MPM (Hu et al. 2018) in the compact form of Taichi's mpm88, in 3D. Each
// substep scatters particles to a grid (mass, momentum, and a weakly compressible pressure from how far each
// particle's volume J has drifted from rest), updates the grid under radial gravity, then gathers velocities and
// their APIC gradients back. This is the CPU reference; materials/oceanFluid.ts runs the same step on the GPU.
// Lengths are orb radii, in the anchor's frame.

export const FLUID = {
  /** Half-width of the simulation box: the cube's corners reach 1.24, and spray rises above them. */
  extent: 1.5,
  /** Radial gravity on splashes, orb radii per second squared (the old spray's). */
  gravity: 1.3,
  /** Bulk stiffness; its square root is the speed of sound, which the substep must respect. */
  stiffness: 9,
  /** Air drag on droplets and on mist, per second. */
  drag: 0.45,
  mistDrag: 1.3,
  /** Longest substep, seconds: CFL for sound and spray speeds at the finest cell. */
  maxStep: 1 / 240,
  /** Fixed-point scale for the grid's atomic sums (WebGPU has no float atomics). */
  fixed: 2 ** 20,
  /**
   * Sea water one splash particle carries: what it takes from the vertex it leaves and gives back where it
   * lands, about 0.08 mm of height on the desktop mesh. Kept tiny on purpose: landings raise the bank, a rising
   * bank reads as surf arriving, and surf throws more spray, so a bigger share feeds back into a runaway sea.
   * The sim keeps the sea's volume exact on its own.
   */
  splashVolume: 2e-8,
  /** Fixed-point scales for the landings the sim adds up per vertex: volume, and foam. */
  landing: { volume: 1e10, foam: 1e6 },
  /** Water is allowed to compress but never to pull: free droplets fly apart rather than clumping. */
  minJ: 0.5,
  maxJ: 1,
} as const;

export interface FluidGrid {
  /** Cells per side. */
  res: number;
  dx: number;
  invDx: number;
}

export function fluidGrid(cell: number): FluidGrid {
  const res = Math.ceil((2 * FLUID.extent) / cell);
  const dx = (2 * FLUID.extent) / res;
  return { res, dx, invDx: 1 / dx };
}

/** Quadratic B-spline weights for the three cells from base, where f = x / dx - base lies in [0.5, 1.5). */
export function splineWeights(f: number): [number, number, number] {
  return [0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2];
}

/** A particle's volume and mass at rest: two particles per cell side. */
export function particleVolume(grid: FluidGrid): number {
  return (grid.dx * 0.5) ** 3;
}

export function toFixed(x: number): number {
  return Math.round(x * FLUID.fixed);
}

export function fromFixed(x: number): number {
  return x / FLUID.fixed;
}

export interface FluidState {
  /** Positions and velocities, three per particle. */
  x: Float64Array;
  v: Float64Array;
  /** APIC velocity gradient, nine per particle, row-major. */
  C: Float64Array;
  /** Volume ratio to rest. */
  J: Float64Array;
}

export function createFluidState(count: number): FluidState {
  return {
    x: new Float64Array(count * 3),
    v: new Float64Array(count * 3),
    C: new Float64Array(count * 9),
    J: new Float64Array(count).fill(1),
  };
}

/** One substep of dt for every particle, on a dense grid; `gravity` 0 turns gravity off. */
export function stepFluid(
  state: FluidState,
  grid: FluidGrid,
  dt: number,
  gravity: number = FLUID.gravity,
): void {
  const { res, dx, invDx } = grid;
  const count = state.J.length;
  const mass = new Float64Array(res * res * res);
  const momentum = new Float64Array(res * res * res * 3);
  const volume = particleVolume(grid);
  const index = (i: number, j: number, k: number) => (k * res + j) * res + i;

  const neighbourhood = (p: number) => {
    const base: number[] = [];
    const weights: [number, number, number][] = [];
    for (let d = 0; d < 3; d++) {
      const g = (state.x[p * 3 + d]! + FLUID.extent) * invDx;
      const b = Math.floor(g - 0.5);
      base.push(b);
      weights.push(splineWeights(g - b));
    }
    return { base, weights };
  };

  // particles to grid
  for (let p = 0; p < count; p++) {
    const { base, weights } = neighbourhood(p);
    const stress = -dt * 4 * FLUID.stiffness * volume * (state.J[p]! - 1) * invDx * invDx;
    for (let a = 0; a < 3; a++)
      for (let b = 0; b < 3; b++)
        for (let c = 0; c < 3; c++) {
          const w = weights[0]![a]! * weights[1]![b]! * weights[2]![c]!;
          const dpos = [
            (base[0]! + a) * dx - FLUID.extent - state.x[p * 3]!,
            (base[1]! + b) * dx - FLUID.extent - state.x[p * 3 + 1]!,
            (base[2]! + c) * dx - FLUID.extent - state.x[p * 3 + 2]!,
          ];
          const cell = index(base[0]! + a, base[1]! + b, base[2]! + c);
          mass[cell]! += w * volume;
          for (let r = 0; r < 3; r++) {
            let affine = stress * dpos[r]!;
            for (let s = 0; s < 3; s++) affine += volume * state.C[p * 9 + r * 3 + s]! * dpos[s]!;
            momentum[cell * 3 + r]! += w * (volume * state.v[p * 3 + r]! + affine);
          }
        }
  }

  // grid: velocities, radial gravity, nothing leaves the box
  const velocity = new Float64Array(res * res * res * 3);
  for (let k = 0; k < res; k++)
    for (let j = 0; j < res; j++)
      for (let i = 0; i < res; i++) {
        const cell = index(i, j, k);
        if (mass[cell]! <= 0) continue;
        const at = [i * dx - FLUID.extent, j * dx - FLUID.extent, k * dx - FLUID.extent];
        const r = Math.hypot(at[0]!, at[1]!, at[2]!) || 1;
        const edge = i < 2 || j < 2 || k < 2 || i >= res - 2 || j >= res - 2 || k >= res - 2;
        for (let d = 0; d < 3; d++)
          velocity[cell * 3 + d] = edge
            ? 0
            : momentum[cell * 3 + d]! / mass[cell]! - (dt * gravity * at[d]!) / r;
      }

  // grid to particles
  for (let p = 0; p < count; p++) {
    const { base, weights } = neighbourhood(p);
    const v = [0, 0, 0];
    const C = new Array<number>(9).fill(0);
    for (let a = 0; a < 3; a++)
      for (let b = 0; b < 3; b++)
        for (let c = 0; c < 3; c++) {
          const w = weights[0]![a]! * weights[1]![b]! * weights[2]![c]!;
          const dpos = [
            (base[0]! + a) * dx - FLUID.extent - state.x[p * 3]!,
            (base[1]! + b) * dx - FLUID.extent - state.x[p * 3 + 1]!,
            (base[2]! + c) * dx - FLUID.extent - state.x[p * 3 + 2]!,
          ];
          const cell = index(base[0]! + a, base[1]! + b, base[2]! + c);
          for (let r = 0; r < 3; r++) {
            const g = velocity[cell * 3 + r]!;
            v[r]! += w * g;
            for (let s = 0; s < 3; s++) C[r * 3 + s]! += 4 * invDx * invDx * w * g * dpos[s]!;
          }
        }
    const trace = C[0]! + C[4]! + C[8]!;
    state.J[p] = Math.min(FLUID.maxJ, Math.max(FLUID.minJ, state.J[p]! * (1 + dt * trace)));
    for (let d = 0; d < 3; d++) {
      state.v[p * 3 + d] = v[d]!;
      state.x[p * 3 + d]! += dt * v[d]!;
    }
    state.C.set(C, p * 9);
  }
}
