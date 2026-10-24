/** Fur tuning lives here; coat generation, the GPU solver and shaders all share these defaults. */
export const FUR = {
  /** Simulated guide length; visible fibre lengths are set by undercoat and guard below. */
  length: 0.24,
  density: 128,
  touchDensity: 72,
  undercoat: { share: 0.65, min: 0.05, max: 0.15 },
  guard: { min: 0.12, max: 0.24 },
  /** Only guides are simulated; the visible coat interpolates their movement. */
  guides: (coarse: boolean) => (coarse ? 1536 : 4096),
  /** Ribbon segments per strand; more makes bent hair curve more smoothly. */
  segments: (coarse: boolean) => (coarse ? 6 : 10),
  /** Simulated points per guide, root included; visible fibres follow their blended curves. */
  points: (coarse: boolean) => (coarse ? 6 : 8),
  seed: 0xc08f1,
  rootSeparation: 1.65,
  widthScale: 0.63,
  colors: {
    root: '#b6804d',
    tip: '#f3b97f',
    copper: '#c76b36',
    rust: '#704324',
  },
  /** Key light direction in view space. */
  light: [-0.55, 0.75, 0.5],
  groom: {
    direction: [0.22, -1, 0.35],
    xWave: { y: 5, z: 3, amplitude: 0.12 },
    zWave: { x: 5, amplitude: 0.1 },
    rootLean: 2.25,
    bend: 3.0,
  },
  physics: {
    rate: 240,
    maxSteps: 10,
    gravity: 2.5,
    springRoot: 4400,
    springTip: 280,
    dampingRatio: 0.65,
    inertia: 0.25,
    maxStep: 0.5,
    brushDrag: 18,
    groomRelax: 0.12,
    combLean: 3.2,
    skin: 0.0015,
  },
} as const;

/** Visible fibres have their own roots and lengths; only the much smaller set of guides has physics. */
export interface FurCoat {
  count: number;
  guideCount: number;
  /** xyz unit direction, w fibre length. */
  roots: Float32Array;
  /** Width variation, pigment variation, brightness variation, guard-hair flag. */
  traits: Float32Array;
  /** Three nearest guide indices and their normalized inverse-distance weights (vec4-aligned). */
  guides: Uint32Array;
  weights: Float32Array;
  /** xyz unit direction, w unused. Every guide spans the full coat length. */
  guideRoots: Float32Array;
}

/** Bound the integration interval and preserve the velocity represented by Verlet's position history. */
export function furFrame(dt: number, previousStep: number) {
  if (!(dt > 0) || !Number.isFinite(dt)) return null;
  const { rate, maxSteps } = FUR.physics;
  const elapsed = Math.min(dt, maxSteps / rate);
  const steps = Math.min(maxSteps, Math.max(1, Math.ceil(elapsed * rate)));
  const step = elapsed / steps;
  return { dt: elapsed, steps, step, historyScale: step / previousStep };
}

/** The coat lies generally downwards, with a gentle sweep and small, spatially coherent changes of lay. */
export function furGroom(p: ArrayLike<number>, normal: ArrayLike<number>): [number, number, number] {
  const { direction, xWave, zWave } = FUR.groom;
  const flow = [
    direction[0] + xWave.amplitude * Math.sin(p[1]! * xWave.y + p[2]! * xWave.z),
    direction[1],
    direction[2] + zWave.amplitude * Math.sin(p[0]! * zWave.x),
  ];
  const radial = flow[0]! * normal[0]! + flow[1]! * normal[1]! + flow[2]! * normal[2]!;
  // Do not normalize the tangent: its magnitude fades smoothly to zero at a cowlick, including at a pole.
  return [flow[0]! - radial * normal[0]!, flow[1]! - radial * normal[1]!, flow[2]! - radial * normal[2]!];
}

function random(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function directions(count: number, next: () => number): Float32Array {
  const roots = new Float32Array(count * 4);
  // Poisson rejection removes the visible rows of a Fibonacci spiral without leaving arbitrary bald spots.
  const separation = FUR.rootSeparation / Math.sqrt(count);
  const bins = Math.max(1, Math.floor(2 / separation));
  const bin = (v: number) => Math.min(bins - 1, Math.floor(((v + 1) / 2) * bins));
  const buckets = new Map<number, number[]>();
  let i = 0;
  while (i < count) {
    const y = next() * 2 - 1;
    const r = Math.sqrt(1 - y * y);
    const phi = next() * Math.PI * 2;
    const x = Math.cos(phi) * r;
    const z = Math.sin(phi) * r;
    const bx = bin(x),
      by = bin(y),
      bz = bin(z);
    let close = false;
    for (let a = Math.max(0, bx - 1); a <= Math.min(bins - 1, bx + 1) && !close; a++)
      for (let b = Math.max(0, by - 1); b <= Math.min(bins - 1, by + 1) && !close; b++)
        for (let c = Math.max(0, bz - 1); c <= Math.min(bins - 1, bz + 1) && !close; c++)
          for (const j of buckets.get((a * bins + b) * bins + c) ?? [])
            if (
              (roots[j * 4]! - x) ** 2 + (roots[j * 4 + 1]! - y) ** 2 + (roots[j * 4 + 2]! - z) ** 2 <
              separation ** 2
            ) {
              close = true;
              break;
            }
    if (close) continue;
    roots.set([x, y, z, 0], i * 4);
    const key = (bx * bins + by) * bins + bz;
    let list = buckets.get(key);
    if (!list) buckets.set(key, (list = []));
    list.push(i++);
  }
  return roots;
}

interface GuideTree {
  index: number;
  axis: number;
  left: GuideTree | null;
  right: GuideTree | null;
}

/** A small kd-tree makes construction O(hairs log guides), instead of comparing every hair to every guide. */
function guideTree(roots: Float32Array, indices: number[], depth = 0): GuideTree | null {
  if (!indices.length) return null;
  const axis = depth % 3;
  indices.sort((a, b) => roots[a * 4 + axis]! - roots[b * 4 + axis]!);
  const middle = indices.length >> 1;
  return {
    index: indices[middle]!,
    axis,
    left: guideTree(roots, indices.slice(0, middle), depth + 1),
    right: guideTree(roots, indices.slice(middle + 1), depth + 1),
  };
}

/** Seeded generation keeps tufts and colour variations attached to the object across resets. */
export function createFurCoat(count: number, guideCount: number, seed: number = FUR.seed): FurCoat {
  if (guideCount < 3) throw new Error('A coat needs at least three guides');
  const next = random(seed);
  const guideRoots = directions(guideCount, next);
  const roots = directions(count, next);
  const traits = new Float32Array(count * 4);
  const guides = new Uint32Array(count * 4);
  const weights = new Float32Array(count * 4);
  const tree = guideTree(
    guideRoots,
    Array.from({ length: guideCount }, (_, i) => i),
  )!;
  const layers = new Uint8Array(count);
  layers.fill(1, 0, count - Math.round(count * FUR.undercoat.share));
  for (let i = count - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [layers[i], layers[j]] = [layers[j]!, layers[i]!];
  }
  const best = [0, 0, 0];
  const distances = [Infinity, Infinity, Infinity];
  let x = 0;
  let y = 0;
  let z = 0;
  const nearest = (node: GuideTree | null): void => {
    if (!node) return;
    const at = node.index * 4;
    const d = (guideRoots[at]! - x) ** 2 + (guideRoots[at + 1]! - y) ** 2 + (guideRoots[at + 2]! - z) ** 2;
    if (d < distances[2]!) {
      let slot = 2;
      while (slot > 0 && d < distances[slot - 1]!) {
        distances[slot] = distances[slot - 1]!;
        best[slot] = best[slot - 1]!;
        slot--;
      }
      distances[slot] = d;
      best[slot] = node.index;
    }
    const delta = [x, y, z][node.axis]! - guideRoots[at + node.axis]!;
    nearest(delta < 0 ? node.left : node.right);
    if (delta * delta <= distances[2]!) nearest(delta < 0 ? node.right : node.left);
  };
  for (let i = 0; i < count; i++) {
    const at = i * 4;
    [x, y, z] = [roots[at]!, roots[at + 1]!, roots[at + 2]!];
    distances.fill(Infinity);
    nearest(tree);
    let total = 0;
    for (let j = 0; j < 3; j++) {
      guides[at + j] = best[j]!;
      weights[at + j] = 1 / Math.max(distances[j]!, 1e-6);
      total += weights[at + j]!;
    }
    for (let j = 0; j < 3; j++) weights[at + j]! /= total;
    const guard = layers[i] === 1;
    const range = guard ? FUR.guard : FUR.undercoat;
    roots[at + 3] = range.min + (range.max - range.min) * next();
    traits.set([next(), next(), next(), guard ? 1 : 0], at);
  }
  return { count, guideCount, roots, traits, guides, weights, guideRoots };
}
