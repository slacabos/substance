import { Vector4, type BufferGeometry } from 'three';
import type { ComputeNode, Node, StorageBufferNode, UniformArrayNode, WebGPURenderer } from 'three/webgpu';
import {
  abs,
  atomicLoad,
  atomicStore,
  cross,
  dot,
  exp,
  float,
  Fn,
  If,
  instancedArray,
  instanceIndex,
  int,
  Loop,
  max,
  min,
  select,
  sign,
  sqrt,
  storage,
  tan,
  uniform,
  uniformArray,
  vec3,
  vec4,
} from 'three/tsl';
import { buildDirectionLookup } from '../core/oceanLookup';
import { oceanMesh } from '../core/oceanMesh';
import { ploughFootprint, SAND, sandHat, sandMesh, type SandMesh } from '../core/sand';

type Vec4Buffer = StorageBufferNode<'vec4'>;
type Float = Node<'float'>;

/** Fixed-point scale for the landings: volume (orb radii cubed) per unit. */
export const LANDING = 1e11;
/** Hand events waiting for the next update: digs and ploughs. */
const EVENTS = 4;
/** Relaxation passes a frame; even, so the bed always ends in the same buffer. */
const RELAX = 2;
/** How long disturbed sand takes to settle back to the look of undisturbed sand, seconds. */
const SETTLE_TIME = 6;

/** A dig (at dir, `depth` deep) or a plough (at dir, moving along `motion`, `amount` high). */
export interface SandEvent {
  dir: readonly [number, number, number];
  amount: number;
  sharpness: number;
  /** Set for a plough: the unit tangent the hand moves along. */
  motion?: readonly [number, number, number];
}

const degrees = (d: number) => (d * Math.PI) / 180;

/**
 * A sand bed on the GPU over an indexed icosphere's vertices, following core/sand.ts step for step: the hand
 * digs and ploughs it, and every frame it relaxes toward its angle of repose (gather-only passes, each vertex
 * summing the pairwise slides with its six neighbours, so it stays conservative without atomics). Rendering
 * reads `renderBuffers()` by vertex index.
 */
export class SandField {
  readonly mesh: SandMesh;
  readonly count: number;
  private readonly renderer: WebGPURenderer;
  private readonly events = Array.from({ length: EVENTS }, () => new Vector4(0, 0, 1, 0));
  private readonly eventShapes = Array.from({ length: EVENTS }, () => new Vector4());
  private pending = 0;
  private readonly dt = uniform(0);
  /** What the frame's footprints didn't quite cancel, taken back evenly (as `reshape` in core/sand.ts). */
  private readonly mean = uniform(0);
  /** 0 to 1: a shake (a hard spin) lowers the repose angle for a moment, setting slopes near it sliding. */
  private readonly shake = uniform(0);
  /** Bed height, wetness, how disturbed it looks (1 fresh to 0 settled); ping-ponged by the relaxation. */
  private readonly beds: [Vec4Buffer, Vec4Buffer];
  /** The bed's gradient on the unit sphere, for normals. */
  private readonly slope: Vec4Buffer;
  /** Unit positions and areas, neighbours, and the nearest-vertex cube map, for things that land on the sand. */
  readonly vertex: Vec4Buffer;
  readonly neighbor: StorageBufferNode<'uint'>;
  readonly lookup: StorageBufferNode<'uint'>;
  /**
   * Sand thrown off or landed on each vertex since the last update, in fixed point (LANDING volume per unit):
   * flying grains (materials/sandGrains.ts) add to it atomically, the next update spends it on the bed.
   */
  readonly landings: StorageBufferNode<'int'>;
  private readonly landNode: ComputeNode;
  private readonly handNode: ComputeNode;
  private readonly relaxNodes: [ComputeNode, ComputeNode];
  private readonly slopeNode: ComputeNode;
  private readonly resetNode: ComputeNode;

  constructor(renderer: WebGPURenderer, geometry: BufferGeometry) {
    this.renderer = renderer;
    const mesh = sandMesh(oceanMesh(geometry));
    this.mesh = mesh;
    const n = mesh.areas.length;
    this.count = n;
    const vertex = instancedArray(
      Float32Array.from({ length: n * 4 }, (_, k) =>
        k % 4 < 3 ? mesh.positions[Math.floor(k / 4) * 3 + (k % 4)]! : mesh.areas[Math.floor(k / 4)]!,
      ),
      'vec4',
    ) as Vec4Buffer;
    const neighbor = instancedArray(mesh.neighbors, 'uint') as StorageBufferNode<'uint'>;
    this.vertex = vertex;
    this.neighbor = neighbor;
    this.lookup = instancedArray(buildDirectionLookup(mesh.positions), 'uint') as StorageBufferNode<'uint'>;
    this.landings = instancedArray(n, 'int').toAtomic() as StorageBufferNode<'int'>;
    this.beds = [instancedArray(n, 'vec4') as Vec4Buffer, instancedArray(n, 'vec4') as Vec4Buffer];
    this.slope = instancedArray(n, 'vec4') as Vec4Buffer;
    const events = uniformArray(this.events, 'vec4') as UniformArrayNode<'vec4'>;
    const shapes = uniformArray(this.eventShapes, 'vec4') as UniformArrayNode<'vec4'>;

    // the hand: digs (a hat footprint) and ploughs (odd along the motion), less the frame's mean
    this.handNode = Fn(() => {
      const i = instanceIndex;
      const q = vertex.element(i).xyz;
      const state = this.beds[0].element(i);
      const change = float(0).toVar();
      for (let e = 0; e < EVENTS; e++) {
        const event = events.element(e);
        const shape = shapes.element(e);
        const r = float(1).sub(dot(q, event.xyz)).mul(shape.w);
        If(event.w.notEqual(0).and(r.lessThan(14)), () => {
          const plough = dot(shape.xyz, shape.xyz).greaterThan(0.5);
          // ploughFootprint (core/sand.ts): a trench with banks either side, a little shoved ahead
          const width = sqrt(shape.w.mul(2));
          const along = dot(q, shape.xyz).mul(width);
          const across = dot(q, cross(event.xyz, shape.xyz)).mul(width);
          const furrow = across.mul(across).sub(1).add(along.mul(0.3)).mul(exp(r.negate())).mul(event.w);
          const dig = event.w.negate().mul(float(1).sub(r)).mul(exp(r.negate()));
          change.addAssign(select(plough, furrow, dig));
        });
      }
      const moved = change.sub(this.mean);
      const bed = max(state.x.add(moved), -SAND.layer);
      // freshly moved sand looks it
      const disturbed = min(state.z.add(abs(moved).mul(400)), 1);
      state.assign(vec4(bed, state.y, disturbed, 0));
    })().compute(n);

    // slideBetween (core/sand.ts), gathered from each end
    const reposeTan = (w: Float) => {
      const { dry, damp, submerged } = SAND.repose;
      const angle = select(
        w.lessThan(0.5),
        float(degrees(dry)).add(w.div(0.5).mul(degrees(damp - dry))),
        float(degrees(damp)).add(
          w
            .sub(0.5)
            .div(0.5)
            .mul(degrees(submerged - damp)),
        ),
      );
      return tan(angle.mul(float(1).sub(this.shake.mul(0.35))));
    };
    const relax = (from: number) =>
      Fn(() => {
        const i = instanceIndex;
        const here = this.beds[from]!.element(i);
        const v = vertex.element(i);
        const change = float(0).toVar();
        Loop({ start: 0, end: 6, type: 'uint' }, ({ i: k }) => {
          const j = neighbor.element(i.mul(6).add(k));
          If(j.notEqual(i), () => {
            const there = this.beds[from]!.element(j);
            const w = vertex.element(j);
            const chord = w.xyz.sub(v.xyz);
            const drop = there.x.sub(here.x);
            const excess = abs(drop).sub(
              sqrt(dot(chord, chord)).mul(reposeTan(here.y.add(there.y).mul(0.5))),
            );
            If(excess.greaterThan(0), () => {
              change.addAssign(
                sign(drop)
                  .mul(SAND.settle)
                  .mul(excess)
                  .mul(w.w.div(v.w.add(w.w))),
              );
            });
          });
        });
        const settled = here.z.mul(exp(this.dt.mul(-1 / (SETTLE_TIME * RELAX))));
        const disturbed = min(settled.add(abs(change).mul(400)), 1);
        this.beds[1 - from]!.element(i).assign(vec4(here.x.add(change), here.y, disturbed, 0));
      })().compute(n);
    this.relaxNodes = [relax(0), relax(1)];

    // the gradient for normals: for a hex ring, sum of (b_j - b_i) d_j over 3 s^2 (twice over sum |d|^2)
    this.slopeNode = Fn(() => {
      const i = instanceIndex;
      const q = vertex.element(i).xyz;
      const b = this.beds[0].element(i).x;
      const sum = vec3(0).toVar();
      const lengths = float(0).toVar();
      Loop({ start: 0, end: 6, type: 'uint' }, ({ i: k }) => {
        const j = neighbor.element(i.mul(6).add(k));
        If(j.notEqual(i), () => {
          const d = vertex.element(j).xyz.sub(q);
          sum.addAssign(d.mul(this.beds[0].element(j).x.sub(b)));
          lengths.addAssign(dot(d, d));
        });
      });
      const g = sum.mul(2).div(max(lengths, 1e-9));
      this.slope.element(i).assign(vec4(g.sub(q.mul(dot(g, q))), 0));
    })().compute(n);

    // grains landed (or thrown off): their volume over the vertex's area, raising (or lowering) its bed
    this.landNode = Fn(() => {
      const i = instanceIndex;
      const volume = float(atomicLoad(this.landings.element(i))).div(LANDING);
      atomicStore(this.landings.element(i), int(0));
      const state = this.beds[0].element(i);
      state.x.assign(max(state.x.add(volume.div(vertex.element(i).w)), -SAND.layer));
    })().compute(n);

    this.resetNode = Fn(() => {
      atomicStore(this.landings.element(instanceIndex), int(0));
      this.beds[0].element(instanceIndex).assign(vec4(0));
      this.beds[1].element(instanceIndex).assign(vec4(0));
      this.slope.element(instanceIndex).assign(vec4(0));
    })().compute(n);
  }

  /** Read-only views for rendering, by vertex index: the bed (height, wetness, disturbance) and its gradient. */
  renderBuffers() {
    return {
      bed: storage(this.beds[0].value, 'vec4', this.count).toReadOnly() as Vec4Buffer,
      slope: storage(this.slope.value, 'vec4', this.count).toReadOnly() as Vec4Buffer,
    };
  }

  /** Queues a dig or plough for the next update; the oldest waiting one gives way. */
  add(event: SandEvent): void {
    const slot = this.pending++ % EVENTS;
    this.events[slot]!.set(event.dir[0], event.dir[1], event.dir[2], event.amount);
    const m = event.motion ?? [0, 0, 0];
    this.eventShapes[slot]!.set(m[0], m[1], m[2], event.sharpness);
  }

  /** A shake from 0 to 1, easing back over a second or so. */
  setShake(amount: number): void {
    this.shake.value = Math.max(0, Math.min(1, amount));
  }

  reset(): void {
    this.renderer.compute(this.resetNode);
  }

  update(dt: number): void {
    if (dt <= 0) return;
    this.dt.value = dt;
    const passes: ComputeNode[] = [];
    passes.push(this.landNode);
    if (this.pending > 0) {
      this.mean.value = this.footprintMean();
      passes.push(this.handNode);
    }
    for (let k = 0; k < RELAX; k++) passes.push(this.relaxNodes[k % 2]!);
    passes.push(this.slopeNode);
    this.renderer.compute(passes);
    for (const event of this.events) event.w = 0;
    this.pending = 0;
    this.shake.value *= Math.exp(-dt / 0.6);
  }

  /** The area-weighted mean of this frame's footprints, which the hand pass takes back so no sand is made. */
  private footprintMean(): number {
    const { positions: p, areas } = this.mesh;
    let total = 0;
    let area = 0;
    for (let i = 0; i < this.count; i++) {
      const [x, y, z] = [p[i * 3]!, p[i * 3 + 1]!, p[i * 3 + 2]!];
      area += areas[i]!;
      for (let e = 0; e < EVENTS; e++) {
        const event = this.events[e]!;
        if (event.w === 0) continue;
        const shape = this.eventShapes[e]!;
        const r = (1 - (x * event.x + y * event.y + z * event.z)) * shape.w;
        if (r >= 14) continue;
        const plough = shape.x * shape.x + shape.y * shape.y + shape.z * shape.z > 0.5;
        const d = [event.x, event.y, event.z] as const;
        const m = [shape.x, shape.y, shape.z] as const;
        total +=
          areas[i]! * (plough ? event.w * ploughFootprint(x, y, z, d, m, shape.w, r) : -event.w * sandHat(r));
      }
    }
    return total / area;
  }
}
