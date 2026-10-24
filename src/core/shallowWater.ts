import { OCEAN_DEPTH } from './oceanFFT';
import { WAVE_SLOTS, type WaveKernel } from './waveKernel';

// The interactive sea surface, one step at a time. Two models share the height field:
// - iWave (Tessendorf 2004) with the finite-depth operator from waveKernel.ts: linear and dispersive, which is
//   right wherever the waves are small next to the depth;
// - nonlinear shallow-water equations, h_t = -div((H + h) u) and u_t = -g grad h, which take over in the
//   shallows, where a wave's height is a fair share of the depth and it steepens, surges and runs up.
// They're blended per vertex by depth. This is the CPU reference; the GPU sim (materials/oceanSim.ts) runs
// the same step. Fluxes go edge by edge (what leaves one vertex enters its neighbour), so shallow water is
// conserved exactly and a dry neighbour is simply a wall.

export const SURFACE = {
  /**
   * Gravity for the waves a hand makes, stronger than the FFT swell's (OCEAN_GRAVITY): no gravity wave on an
   * orb this small can outrun the swell by more than about 2x, and a tap's long ring should run like a tsunami,
   * at sqrt(g H) = 0.25 orb radii per second over the open sea, with ripples trailing behind it.
   */
  gravity: 1.25,
  /** iWave's velocity damping, per second: a wave loses half its height in about 6 s. */
  damping: 0.12,
  /** Drag on the surface current, per second. */
  drag: 0.35,
  /** Cap on the current, orb radii per second, so violent drags stay well behaved. */
  maxCurrent: 0.3,
  /** Cap on the height, orb radii. */
  maxHeight: 0.09,
  /**
   * Share of the difference from its neighbours' vertical velocity a vertex loses each step. Real water damps
   * short waves far faster than long ones (viscosity goes as k^2); this does the same, and clears grid-scale
   * chatter in a second or two while long waves barely notice.
   */
  viscosity: 0.03,
  /**
   * The same for the shallow-water update, which otherwise has no damping at all: a share of each vertex's
   * difference from its neighbours (scaled by their mean area over its own) moves to them each step, as fluxes
   * between pairs, so no water is made or lost. Without it a grid-scale ripple can build on a shelf several
   * cells wide.
   */
  shallowSmoothing: 0.01,
  /**
   * How far a hand can press the open sea, orb radii. It reaches less over shallower water (in proportion to
   * the depth), so it can't press the shallows below their bed; a push fades out as the water nears it.
   */
  handReach: 0.09,
} as const;

/** How much of a push the water at height h over still depth `depth` still takes (see SURFACE.handReach). */
export function pushRoom(push: number, h: number, depth: number): number {
  const reach = (SURFACE.handReach * depth) / OCEAN_DEPTH;
  return Math.min(1, Math.max(0, 1 - (h * Math.sign(push)) / reach));
}

/**
 * The lowest the surface may go: the bed, where the shallow-water equations need a water column; over the open
 * sea only the height cap, since there the depth just sets how waves travel.
 */
export function surfaceFloor(depth: number): number {
  const s = shallowWeight(depth);
  return -(SURFACE.maxHeight * (1 - s) + depth * s);
}

/** How much of a vertex follows the shallow-water equations rather than iWave, by still-water depth. */
export function shallowWeight(depth: number): number {
  const t = Math.min(1, Math.max(0, (depth - 0.008) / (0.025 - 0.008)));
  return 1 - t * t * (3 - 2 * t);
}

export interface SurfaceMesh {
  /** Unit-sphere positions. */
  positions: Float32Array;
  /** Six neighbours per vertex; five-fold vertices repeat themselves. */
  neighbors: Uint32Array;
}

export interface SurfaceField {
  /** Still-water depth per vertex. */
  depth: Float32Array;
  /** 1 for water, 0 for land. */
  wet: Uint8Array;
  kernel: WaveKernel;
  /** Area each vertex stands for (see vertexAreas). */
  areas: Float32Array;
}

export interface SurfaceState {
  /** Height now and a step ago, per vertex. */
  h: Float32Array;
  hPrev: Float32Array;
  /** Tangential current, three per vertex. */
  u: Float32Array;
}

export function createSurfaceState(count: number): SurfaceState {
  return { h: new Float32Array(count), hPrev: new Float32Array(count), u: new Float32Array(count * 3) };
}

/** Area each vertex stands for, from its mean edge length (a hexagon of that spacing). */
export function vertexAreas(mesh: SurfaceMesh): Float32Array {
  const { positions: p, neighbors } = mesh;
  const count = p.length / 3;
  const areas = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    let sum = 0;
    let edges = 0;
    for (let k = 0; k < 6; k++) {
      const j = neighbors[i * 6 + k]!;
      if (j === i) continue;
      sum += Math.hypot(p[i * 3]! - p[j * 3]!, p[i * 3 + 1]! - p[j * 3 + 1]!, p[i * 3 + 2]! - p[j * 3 + 2]!);
      edges++;
    }
    const spacing = sum / edges;
    areas[i] = ((Math.sqrt(3) / 2) * spacing * spacing * edges) / 6;
  }
  return areas;
}

/**
 * Advances the surface by dt. `accel[i]` pushes vertex i's height (orb radii per second squared): a finger's
 * pressure, or an impulse spread over one step. Writes the new state in place.
 */
export function stepSurface(
  mesh: SurfaceMesh,
  field: SurfaceField,
  state: SurfaceState,
  dt: number,
  accel?: ArrayLike<number>,
): void {
  const { positions: p, neighbors } = mesh;
  const { depth, wet, kernel } = field;
  const { h, hPrev, u } = state;
  const count = h.length;
  const g = SURFACE.gravity;
  const nextH = new Float32Array(count);
  const nextU = new Float32Array(count * 3);
  const areas = field.areas;
  let meanArea = 0;
  for (let i = 0; i < count; i++) meanArea += areas[i]! / count;
  const c = [0, 0, 0];

  // Chord from i to j: unit direction into c, length returned. Its face in the dual hexagon is length/sqrt 3
  // long; using the chord (the same from both ends) is what makes each edge's flux cancel exactly.
  const chord = (i: number, j: number) => {
    for (let k = 0; k < 3; k++) c[k] = p[j * 3 + k]! - p[i * 3 + k]!;
    const length = Math.hypot(c[0]!, c[1]!, c[2]!);
    for (let k = 0; k < 3; k++) c[k]! /= length;
    return length;
  };

  // momentum: u += dt (-g grad h - drag u); dry neighbours add no slope
  const drag = Math.exp(-SURFACE.drag * dt);
  for (let i = 0; i < count; i++) {
    if (!wet[i]) continue;
    const gradient = [0, 0, 0];
    for (let k = 0; k < 6; k++) {
      const j = neighbors[i * 6 + k]!;
      if (j === i || !wet[j]) continue;
      const face = chord(i, j) / Math.sqrt(3);
      for (let d = 0; d < 3; d++) gradient[d]! += ((h[j]! - h[i]!) / 2) * c[d]! * face;
    }
    let radial = 0;
    for (let d = 0; d < 3; d++) {
      nextU[i * 3 + d] = (u[i * 3 + d]! - (dt * g * gradient[d]!) / areas[i]!) * drag;
      radial += nextU[i * 3 + d]! * p[i * 3 + d]!;
    }
    let speed = 0;
    for (let d = 0; d < 3; d++) {
      nextU[i * 3 + d]! -= radial * p[i * 3 + d]!;
      speed += nextU[i * 3 + d]! ** 2;
    }
    speed = Math.sqrt(speed);
    if (speed > SURFACE.maxCurrent)
      for (let d = 0; d < 3; d++) nextU[i * 3 + d]! *= SURFACE.maxCurrent / speed;
  }

  // Damping only the vertical velocity, with a central difference: h_tt + a h_t = -g G h. (The often-quoted
  // iWave update also pulls the height itself toward zero, a hidden spring that swamps slow waves at 60 Hz.)
  const a = (SURFACE.damping * dt) / 2;
  for (let i = 0; i < count; i++) {
    if (!wet[i]) continue;
    // iWave, with dry neighbours mirroring this vertex so the coast reflects
    let operator = 0;
    for (let s = 0; s < WAVE_SLOTS; s++) {
      const j = kernel.indices[i * WAVE_SLOTS + s]!;
      operator += kernel.weights[i * WAVE_SLOTS + s]! * (wet[j] ? h[j]! : h[i]!);
    }
    const force = (accel?.[i] ?? 0) * dt * dt;
    const push = force * pushRoom(force, h[i]!, depth[i]!);
    // last step's rise, eased toward the neighbours' (viscosity)
    let neighbourRise = 0;
    let wetNeighbours = 0;
    for (let k = 0; k < 6; k++) {
      const j = neighbors[i * 6 + k]!;
      if (j === i || !wet[j]) continue;
      neighbourRise += h[j]! - hPrev[j]!;
      wetNeighbours++;
    }
    const rise = h[i]! - hPrev[i]!;
    const eased = wetNeighbours ? rise + SURFACE.viscosity * (neighbourRise / wetNeighbours - rise) : rise;
    const previous = h[i]! - eased;
    const wave = (2 * h[i]! - previous * (1 - a) - g * dt * dt * operator + push) / (1 + a);

    // Shallow water: the flux through each face is the edge's mean current times the water column on the side
    // it flows from (upwind), so a cell that has run dry has nothing to give; none goes through a wall.
    let outflow = 0;
    for (let k = 0; k < 6; k++) {
      const j = neighbors[i * 6 + k]!;
      if (j === i || !wet[j]) continue;
      const face = chord(i, j) / Math.sqrt(3);
      let velocity = 0;
      for (let d = 0; d < 3; d++) velocity += (nextU[i * 3 + d]! + nextU[j * 3 + d]!) * 0.5 * c[d]!;
      const column = Math.max(0, velocity > 0 ? depth[i]! + h[i]! : depth[j]! + h[j]!);
      outflow += velocity * column * face;
    }
    let spread = 0;
    for (let k = 0; k < 6; k++) {
      const j = neighbors[i * 6 + k]!;
      if (j !== i && wet[j]) spread += h[j]! - h[i]!;
    }
    const smoothing = (SURFACE.shallowSmoothing * meanArea) / areas[i]!;
    const shallow = h[i]! - (dt * outflow) / areas[i]! + smoothing * spread + push;

    const s = shallowWeight(depth[i]!);
    nextH[i] = Math.min(SURFACE.maxHeight, Math.max(surfaceFloor(depth[i]!), wave * (1 - s) + shallow * s));
  }
  hPrev.set(h);
  h.set(nextH);
  u.set(nextU);
}

/** Total water above or below rest, sum of height times area. */
export function surfaceVolume(state: SurfaceState, areas: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < state.h.length; i++) sum += state.h[i]! * areas[i]!;
  return sum;
}
