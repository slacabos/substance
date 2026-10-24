// The vertical-derivative operator behind iWave (Tessendorf 2004), for water of finite depth H: for a wave
// e^{ik.x} it returns k tanh(kH) times the wave, which gives w^2 = g k tanh(kH). Short waves disperse like deep
// water; long ones run at sqrt(gH) like a tsunami; as H goes to zero it becomes H times the Laplacian, the
// linear shallow-water limit. (A finite stencil can't match deep water's |k|, which has a corner at k = 0;
// k tanh(kH) is smooth, so it can.)
//
// On the ocean's icosphere every vertex sees a near-perfect hexagonal lattice, so the operator is a weighted
// sum over its first four rings of neighbours. Weights are fitted by least squares on an ideal lattice, for a
// table of depths, then applied at each vertex's own depth and neighbour distances.

/** Rings of neighbours the operator reaches; four rings hold 60 neighbours on a hex lattice. */
export const WAVE_RINGS = 4;
/** Slots per vertex: itself, then up to 60 neighbours (unused slots repeat the vertex with weight 0). */
export const WAVE_SLOTS = 61;
/** Deepest water, in spacings, the stencil still fits well (about 5% worst error). */
export const WAVE_MAX_DEPTH = 2;
const DEPTH_STEPS = 32;
/**
 * Ridge penalty on the fitted weights. Unpenalised, the deeper fits alternate in sign and turn negative at
 * some wavenumbers (an unstable sea), and the icosphere's slightly bent hexagons amplify their error; this
 * keeps the operator positive everywhere for a few percent of accuracy.
 */
const RIDGE = 10;

/**
 * An ideal hex lattice within the rings, without the centre: its offsets, its distinct shell distances, and for
 * each ring the shell of every member in order of distance.
 */
export function lattice(): { offsets: [number, number][]; shells: number[]; rings: number[][] } {
  const offsets: [number, number][] = [];
  const steps: number[] = [];
  for (let a = -WAVE_RINGS; a <= WAVE_RINGS; a++) {
    for (let b = -WAVE_RINGS; b <= WAVE_RINGS; b++) {
      // hex distance in lattice steps, which is the breadth-first ring on the mesh
      const ring = Math.max(Math.abs(a), Math.abs(b), Math.abs(a + b));
      if (ring === 0 || ring > WAVE_RINGS) continue;
      offsets.push([a + b / 2, (b * Math.sqrt(3)) / 2]);
      steps.push(ring);
    }
  }
  const radius = (o: [number, number]) => Math.round(Math.hypot(o[0], o[1]) * 1e6) / 1e6;
  const shells = [...new Set(offsets.map(radius))].sort((p, q) => p - q);
  const rings = Array.from({ length: WAVE_RINGS }, (_, r) =>
    offsets
      .filter((_, j) => steps[j] === r + 1)
      .map((o) => shells.indexOf(radius(o)))
      .sort((p, q) => p - q),
  );
  return { offsets, shells, rings };
}

function solve(a: number[][], b: number[]): number[] {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < n; c++) {
    let pivot = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r]![c]!) > Math.abs(m[pivot]![c]!)) pivot = r;
    [m[c], m[pivot]] = [m[pivot]!, m[c]!];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = m[r]![c]! / m[c]![c]!;
      for (let k = c; k <= n; k++) m[r]![k]! -= f * m[c]![k]!;
    }
  }
  return m.map((row, i) => row[n]! / row[i]!);
}

/** The dispersion target the stencil matches, k tanh(kH), for unit spacing. */
export const waveSymbol = (k: number, depth: number): number => k * Math.tanh(k * depth);

/**
 * Shell weights (unit spacing) such that sum_j w(|d_j|) (cos(k.d_j) - 1) ~ k tanh(k depth), in relative terms,
 * for |k| up to half the lattice's Nyquist rate and in many directions, so the operator is close to isotropic.
 * Exported for tests, with the lattice the weights apply to.
 */
export function fitShellWeights(depth: number): { shells: number[]; weights: number[] } {
  const { offsets, shells } = lattice();
  if (depth <= 0) return { shells, weights: shells.map(() => 0) };
  const shellOf = offsets.map(([x, y]) => shells.findIndex((s) => Math.abs(s - Math.hypot(x, y)) < 1e-5));
  const n = shells.length;
  const ata = Array.from({ length: n }, () => Array<number>(n).fill(0));
  const atb = Array<number>(n).fill(0);
  const kMax = (0.5 * (2 * Math.PI)) / Math.sqrt(3);
  for (let i = 1; i <= 30; i++) {
    const k = (kMax * i) / 30;
    const target = waveSymbol(k, depth);
    for (let d = 0; d < 8; d++) {
      const angle = (d / 8) * (Math.PI / 3);
      const kx = k * Math.cos(angle);
      const ky = k * Math.sin(angle);
      const row = shells.map(() => 0);
      offsets.forEach(([x, y], j) => (row[shellOf[j]!]! += (Math.cos(kx * x + ky * y) - 1) / target));
      for (let a = 0; a < n; a++) {
        atb[a]! += row[a]!;
        for (let b = 0; b < n; b++) ata[a]![b]! += row[a]! * row[b]!;
      }
    }
  }
  for (let a = 0; a < n; a++) ata[a]![a]! += RIDGE;
  return { shells, weights: solve(ata, atb) };
}

export interface WaveKernel {
  /** WAVE_SLOTS vertex indices per vertex, itself first. */
  indices: Uint32Array;
  /** Matching weights; they sum to zero, so a flat surface stays flat. Units of 1 / length. */
  weights: Float32Array;
}

/**
 * The operator for every vertex of a mesh (unit-sphere positions, six neighbours each as from oceanMesh), for
 * still-water depth `depth[i]` (same length units). Distances and depth are measured against each vertex's
 * own mean edge length, so the slightly uneven icosphere still gets the right dispersion. Depths beyond what
 * the stencil can hold are treated as WAVE_MAX_DEPTH spacings.
 */
export function buildWaveKernel(
  positions: Float32Array,
  neighbors: Uint32Array,
  depth: ArrayLike<number>,
): WaveKernel {
  const count = positions.length / 3;
  const table = Array.from({ length: DEPTH_STEPS + 1 }, (_, i) =>
    fitShellWeights((WAVE_MAX_DEPTH * i) / DEPTH_STEPS),
  );
  const { rings: latticeRings } = lattice();
  const indices = new Uint32Array(count * WAVE_SLOTS);
  const weights = new Float32Array(count * WAVE_SLOTS);
  const distance = (a: number, b: number) =>
    Math.hypot(
      positions[a * 3]! - positions[b * 3]!,
      positions[a * 3 + 1]! - positions[b * 3 + 1]!,
      positions[a * 3 + 2]! - positions[b * 3 + 2]!,
    );
  // shell weight for depth d (spacings), interpolated between fitted depths
  const weightAt = (d: number, shell: number): number => {
    const t = (Math.min(Math.max(d, 0), WAVE_MAX_DEPTH) / WAVE_MAX_DEPTH) * DEPTH_STEPS;
    const i = Math.min(DEPTH_STEPS - 1, Math.floor(t));
    const f = t - i;
    return table[i]!.weights[shell]! * (1 - f) + table[i + 1]!.weights[shell]! * f;
  };
  const seen = new Int32Array(count).fill(-1);
  for (let i = 0; i < count; i++) {
    // Breadth-first over the neighbour graph. Each ring's members take the lattice ring's shells in order of
    // distance (by share of the ring, since rings next to the twelve five-fold vertices are a little short).
    const members: number[] = [];
    const shellOf: number[] = [];
    seen[i] = i;
    let frontier = [i];
    for (let level = 0; level < WAVE_RINGS; level++) {
      const next: number[] = [];
      for (const v of frontier) {
        for (let k = 0; k < 6; k++) {
          const j = neighbors[v * 6 + k]!;
          if (seen[j] === i) continue;
          seen[j] = i;
          next.push(j);
        }
      }
      next.sort((a, b) => distance(i, a) - distance(i, b));
      const shells = latticeRings[level]!;
      next.forEach((j, q) => {
        members.push(j);
        shellOf.push(
          shells[Math.min(shells.length - 1, Math.floor(((q + 0.5) / next.length) * shells.length))]!,
        );
      });
      frontier = next;
    }
    let local = 0;
    let edges = 0;
    for (let k = 0; k < 6; k++) {
      const j = neighbors[i * 6 + k]!;
      if (j === i) continue;
      local += distance(i, j);
      edges++;
    }
    local /= edges;
    const d = depth[i]! / local;
    const base = i * WAVE_SLOTS;
    let sum = 0;
    indices[base] = i;
    for (let s = 1; s < WAVE_SLOTS; s++) {
      const j = members[s - 1];
      if (j === undefined) {
        indices[base + s] = i;
        continue;
      }
      const w = weightAt(d, shellOf[s - 1]!) / local;
      indices[base + s] = j;
      weights[base + s] = w;
      sum += w;
    }
    weights[base] = -sum;
  }
  return { indices, weights };
}

/** Applies the operator: out[i] = sum of weight * h over vertex i's slots. */
export function applyWaveKernel(kernel: WaveKernel, h: ArrayLike<number>, out: Float32Array): void {
  const count = out.length;
  for (let i = 0; i < count; i++) {
    let sum = 0;
    for (let s = 0; s < WAVE_SLOTS; s++) {
      sum += kernel.weights[i * WAVE_SLOTS + s]! * h[kernel.indices[i * WAVE_SLOTS + s]!]!;
    }
    out[i] = sum;
  }
}
