import { vertexAreas, type SurfaceMesh } from './shallowWater';

// Sand as a height field: a bed of sand over a surface, on the same hex mesh as the sea (core/oceanMesh.ts). Sand
// slides wherever a slope is steeper than its angle of repose, the hand digs and ploughs it, and moving water
// smooths it. Everything moves sand between neighbours in pairs, so none is ever made or lost. This is the CPU
// reference; materials/sandField.ts runs the same steps on the GPU.

export const SAND = {
  /** Angle of repose of dry sand, of damp sand (cohesive), and of sand under water, in degrees. */
  repose: { dry: 33, damp: 45, submerged: 28 },
  /**
   * Share of a pair's excess slope settled per relaxation pass. Every vertex has six neighbours, so above 1/6 a
   * vertex could give away more than its excess and the slope would flip.
   */
  settle: 0.14,
  /** Thickness of the sand layer above its core, orb radii: digging stops there. */
  layer: 0.12,
} as const;

/** tan of the angle of repose for wetness w: 0 dry, 0.5 damp, 1 under water. */
export function reposeSlope(w: number): number {
  const { dry, damp, submerged } = SAND.repose;
  const degrees = w < 0.5 ? dry + (damp - dry) * (w / 0.5) : damp + (submerged - damp) * ((w - 0.5) / 0.5);
  return Math.tan((degrees * Math.PI) / 180);
}

/** The mesh a sand bed lies on: unit positions, six neighbours each, and the area each vertex stands for. */
export interface SandMesh extends SurfaceMesh {
  areas: Float32Array;
}

export function sandMesh(mesh: SurfaceMesh): SandMesh {
  return { ...mesh, areas: vertexAreas(mesh) };
}

export interface SandState {
  /** Height of the sand above its rest surface, per vertex, orb radii (down to -SAND.layer). */
  bed: Float32Array;
  /** 0 dry, 0.5 damp, 1 under water. */
  wet: Float32Array;
}

export function createSandState(count: number): SandState {
  return { bed: new Float32Array(count), wet: new Float32Array(count) };
}

/** Sum of height times area: the sand above (or missing below) the rest surface. */
export function sandVolume(mesh: SandMesh, state: SandState): number {
  let sum = 0;
  for (let i = 0; i < state.bed.length; i++) sum += state.bed[i]! * mesh.areas[i]!;
  return sum;
}

function chord(p: Float32Array, i: number, j: number): number {
  return Math.hypot(p[i * 3]! - p[j * 3]!, p[i * 3 + 1]! - p[j * 3 + 1]!, p[i * 3 + 2]! - p[j * 3 + 2]!);
}

/**
 * Height vertex i gains from neighbour j in one relaxation pass: where the slope between them is steeper than
 * the repose slope (at their mean wetness), the pair settles a share of the excess, the higher giving to the
 * lower. Equal and opposite in volume (heights weighted by the other's area), so sand is conserved; the GPU pass
 * evaluates the same pair from each end.
 */
export function slideBetween(
  bedI: number,
  bedJ: number,
  wetI: number,
  wetJ: number,
  length: number,
  areaI: number,
  areaJ: number,
): number {
  const drop = bedJ - bedI;
  const excess = Math.abs(drop) - length * reposeSlope((wetI + wetJ) / 2);
  if (excess <= 0) return 0;
  return Math.sign(drop) * SAND.settle * excess * (areaJ / (areaI + areaJ));
}

/** One relaxation pass: every vertex gathers what slides to or from each neighbour (Jacobi, as on the GPU). */
export function relaxSand(mesh: SandMesh, state: SandState): void {
  const { positions: p, neighbors, areas } = mesh;
  const { bed, wet } = state;
  const next = new Float32Array(bed.length);
  for (let i = 0; i < bed.length; i++) {
    let change = 0;
    for (let k = 0; k < 6; k++) {
      const j = neighbors[i * 6 + k]!;
      if (j === i) continue;
      change += slideBetween(bed[i]!, bed[j]!, wet[i]!, wet[j]!, chord(p, i, j), areas[i]!, areas[j]!);
    }
    next[i] = bed[i]! + change;
  }
  bed.set(next);
}

/**
 * Moving water smooths the sand under it: each pair evens out by a share of their difference set by how fast
 * the water runs over both (`flow`, 0 where there's none). Conservative like the slides.
 */
export function smoothSand(mesh: SandMesh, state: SandState, flow: ArrayLike<number>, rate: number): void {
  const { neighbors, areas } = mesh;
  const { bed } = state;
  const next = new Float32Array(bed.length);
  for (let i = 0; i < bed.length; i++) {
    let change = 0;
    for (let k = 0; k < 6; k++) {
      const j = neighbors[i * 6 + k]!;
      if (j === i) continue;
      const share = Math.min(1 / 7, rate * Math.min(flow[i]!, flow[j]!));
      change += share * (bed[j]! - bed[i]!) * (areas[j]! / (areas[i]! + areas[j]!));
    }
    next[i] = bed[i]! + change;
  }
  bed.set(next);
}

/**
 * The plough's footprint at (x, y, z), per unit amount, r its (1 - cos) * sharpness from the hand: down along the
 * track, up beyond a footprint's width either side (`across` is the distance across in those widths), and a
 * little shoved ahead (odd along the motion m).
 */
export function ploughFootprint(
  x: number,
  y: number,
  z: number,
  d: readonly [number, number, number],
  m: readonly [number, number, number],
  sharpness: number,
  r: number,
): number {
  const width = Math.sqrt(2 * sharpness);
  const along = (x * m[0] + y * m[1] + z * m[2]) * width;
  // the side: d x m
  const across =
    (x * (d[1] * m[2] - d[2] * m[1]) + y * (d[2] * m[0] - d[0] * m[2]) + z * (d[0] * m[1] - d[1] * m[0])) *
    width;
  return (across * across - 1 + PLOUGH_SHOVE * along) * Math.exp(-r);
}

/** How much of a plough's sand goes ahead of the hand rather than to the sides. */
const PLOUGH_SHOVE = 0.3;

/** The hand's footprint at x = (1 - cos angle) * sharpness: down in the middle, up in a rim, zero in total. */
export const sandHat = (x: number): number => (1 - x) * Math.exp(-x);

/**
 * Adds `change(x, y, z)` to the bed at every vertex (x, y, z its unit position), then takes back evenly what it
 * added in total: a footprint that cancels in theory only nearly does on a mesh, and no sand may be made or lost.
 */
function reshape(
  mesh: SandMesh,
  state: SandState,
  change: (x: number, y: number, z: number) => number,
): void {
  const { positions: p, areas } = mesh;
  const { bed } = state;
  let total = 0;
  let area = 0;
  for (let i = 0; i < bed.length; i++) {
    const c = change(p[i * 3]!, p[i * 3 + 1]!, p[i * 3 + 2]!);
    bed[i]! += c;
    total += c * areas[i]!;
    area += areas[i]!;
  }
  const mean = total / area;
  for (let i = 0; i < bed.length; i++) bed[i]! -= mean;
}

/** Digs at unit direction d: sand pushed from the middle into a rim around it, `depth` deep at the centre. */
export function digSand(
  mesh: SandMesh,
  state: SandState,
  d: readonly [number, number, number],
  depth: number,
  sharpness: number,
): void {
  reshape(mesh, state, (x, y, z) => {
    const r = (1 - (x * d[0] + y * d[1] + z * d[2])) * sharpness;
    return r > 14 ? 0 : -depth * sandHat(r);
  });
}

/**
 * Ploughs along a drag at unit direction d moving along tangent m: a trench along the track with the sand banked
 * on either side, `amount` deep (zero in total across the track), plus a little shoved ahead of the hand.
 */
export function ploughSand(
  mesh: SandMesh,
  state: SandState,
  d: readonly [number, number, number],
  m: readonly [number, number, number],
  amount: number,
  sharpness: number,
): void {
  reshape(mesh, state, (x, y, z) => {
    const r = (1 - (x * d[0] + y * d[1] + z * d[2])) * sharpness;
    if (r > 14) return 0;
    return ploughFootprint(x, y, z, d, m, sharpness, r) * amount;
  });
}
