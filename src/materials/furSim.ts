import { Matrix4, Vector3, type Object3D } from 'three';
import type { ComputeNode, Node, StorageBufferNode, WebGPURenderer } from 'three/webgpu';
import {
  acos,
  clamp,
  dot,
  exp,
  float,
  Fn,
  If,
  instancedArray,
  instanceIndex,
  length,
  Loop,
  max,
  min,
  mix,
  normalize,
  Return,
  select,
  smoothstep,
  sqrt,
  storage,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import { FUR, furFrame } from '../core/fur';
import { furGroom } from '../nodes/furGroom';
import { baseN, baseR } from '../nodes/shape';

const SIM = FUR.physics;

/**
 * A small set of coat guides, not one simulation per visible hair. Each guide bends around a groomed rest
 * curve; rendering blends its displacement onto many rooted fibres. Position and Verlet history are world
 * space, and rest holds the current, uncombed curve transformed into world space.
 */
export class FurSim {
  readonly count: number;
  readonly points: number;
  readonly uniforms = {
    anchor: uniform(new Matrix4()),
    anchorInv: uniform(new Matrix4()),
    anchorPrevInv: uniform(new Matrix4()),
    scale: uniform(1),
    dt: uniform(1 / 60),
    steps: uniform(4, 'int'),
    step: uniform(1 / SIM.rate),
    historyScale: uniform(1),
    brushDir: uniform(new Vector3(0, 0, 1)),
    brushStroke: uniform(new Vector3()),
    brushVel: uniform(new Vector3()),
    brushAmount: uniform(0),
    pressDir: uniform(new Vector3(0, 0, 1)),
    press: uniform(0),
  };

  private readonly renderer: WebGPURenderer;
  private readonly roots: StorageBufferNode<'vec4'>;
  private readonly pos: StorageBufferNode<'vec3'>;
  private readonly prev: StorageBufferNode<'vec3'>;
  private readonly rest: StorageBufferNode<'vec3'>;
  private readonly displacement: StorageBufferNode<'vec3'>;
  private readonly groom: StorageBufferNode<'vec3'>;
  private readonly stepNode: ComputeNode;
  private readonly resetNode: ComputeNode;

  constructor(renderer: WebGPURenderer, roots: Float32Array, points: number) {
    this.renderer = renderer;
    this.count = roots.length / 4;
    this.points = points;
    this.roots = instancedArray(roots, 'vec4');
    this.pos = instancedArray(this.count * points, 'vec3');
    this.prev = instancedArray(this.count * points, 'vec3');
    this.rest = instancedArray(this.count * points, 'vec3');
    this.displacement = instancedArray(this.count * points, 'vec3');
    this.groom = instancedArray(this.count, 'vec3');
    this.stepNode = this.createStep().compute(this.count);
    this.resetNode = this.createReset().compute(this.count);
  }

  renderBuffers() {
    const view = <T extends 'vec3' | 'vec4'>(node: StorageBufferNode<T>, type: T) =>
      storage(node.value, type, node.bufferCount).toReadOnly() as StorageBufferNode<T>;
    return { roots: view(this.roots, 'vec4'), displacement: view(this.displacement, 'vec3') };
  }

  reset(anchor: Object3D): void {
    this.setAnchor(anchor);
    this.uniforms.anchorPrevInv.value.copy(this.uniforms.anchorInv.value);
    this.uniforms.step.value = 1 / SIM.rate;
    this.uniforms.historyScale.value = 1;
    this.uniforms.brushAmount.value = 0;
    this.uniforms.brushVel.value.set(0, 0, 0);
    this.uniforms.press.value = 0;
    this.renderer.compute(this.resetNode);
  }

  update(anchor: Object3D, dt: number): void {
    const frame = furFrame(dt, this.uniforms.step.value);
    if (!frame) return;
    this.setAnchor(anchor);
    this.uniforms.dt.value = frame.dt;
    this.uniforms.steps.value = frame.steps;
    this.uniforms.step.value = frame.step;
    this.uniforms.historyScale.value = frame.historyScale;
    this.renderer.compute(this.stepNode);
  }

  private setAnchor(anchor: Object3D): void {
    this.uniforms.anchorPrevInv.value.copy(this.uniforms.anchorInv.value);
    anchor.updateMatrixWorld();
    this.uniforms.anchor.value.copy(anchor.matrixWorld);
    this.uniforms.anchorInv.value.copy(anchor.matrixWorld).invert();
    this.uniforms.scale.value = anchor.scale.x;
  }

  private strandBase() {
    const p = this.roots.element(instanceIndex).xyz;
    const q = p.mul(baseR(p));
    const n = baseN(p, q);
    const segment = this.uniforms.scale.mul(FUR.length / (this.points - 1));
    return { p, q, n, segment, lay: furGroom(p, n) };
  }

  private toWorld(v: Node<'vec3'>, w: number): Node<'vec3'> {
    return this.uniforms.anchor.mul(vec4(v, w)).xyz;
  }

  private createReset() {
    return Fn(() => {
      If(instanceIndex.greaterThanEqual(this.count), () => Return());
      const { q, n, segment, lay } = this.strandBase();
      const base = instanceIndex.mul(this.points);
      const x = this.toWorld(q, 1).toVar();
      this.pos.element(base).assign(x);
      this.prev.element(base).assign(x);
      this.rest.element(base).assign(x);
      this.displacement.element(base).assign(vec3(0));
      Loop({ start: 1, end: this.points, type: 'uint' }, ({ i }) => {
        const bend = float(i)
          .sub(0.5)
          .div(this.points - 1)
          .mul(FUR.groom.bend)
          .add(FUR.groom.rootLean);
        x.addAssign(normalize(this.toWorld(n.add(lay.mul(bend)), 0)).mul(segment));
        this.pos.element(base.add(i)).assign(x);
        this.prev.element(base.add(i)).assign(x);
        this.rest.element(base.add(i)).assign(x);
        this.displacement.element(base.add(i)).assign(vec3(0));
      });
      this.groom.element(instanceIndex).assign(vec3(0));
    })();
  }

  private createStep() {
    const u = this.uniforms;
    return Fn(() => {
      If(instanceIndex.greaterThanEqual(this.count), () => Return());
      const { p, q, n, segment, lay } = this.strandBase();
      const groom = this.groom.element(instanceIndex);
      const bs = acos(clamp(dot(p, u.brushDir), -1, 1));
      const comb = exp(bs.mul(bs).div(-0.05)).mul(u.brushAmount);
      const combed = mix(groom, u.brushStroke.mul(SIM.combLean), clamp(comb.mul(u.dt).mul(10), 0, 1)).mul(
        exp(u.dt.mul(-SIM.groomRelax)),
      );
      groom.assign(combed.sub(n.mul(dot(combed, n))));
      const away = p.sub(u.pressDir.mul(dot(p, u.pressDir)));
      const ps = acos(clamp(dot(p, u.pressDir), -1, 1));
      const part = away
        .div(max(length(away), 1e-5))
        .mul(u.press.mul(1.1))
        .mul(exp(ps.mul(ps).div(-0.035)))
        .mul(smoothstep(0, 0.08, ps));
      // A brush can replace the natural lay, rather than having to fight an equally strong built-in groom.
      const natural = float(1).sub(clamp(length(groom).mul(0.6), 0, 0.85));
      const lean = lay.mul(natural).add(groom).add(part);
      const brushVel = this.toWorld(u.brushVel, 0);
      const root = this.toWorld(q, 1);
      const base = instanceIndex.mul(this.points);
      this.pos.element(base).assign(root);
      this.prev.element(base).assign(root);
      this.rest.element(base).assign(root);
      this.displacement.element(base).assign(vec3(0));
      const carry = (x: Node<'vec3'>) =>
        mix(x, this.toWorld(u.anchorPrevInv.mul(vec4(x, 1)).xyz, 1), 1 - SIM.inertia);
      const baseline = root.toVar();
      Loop({ start: 1, end: this.points, type: 'uint' }, ({ i }) => {
        const at = base.add(i);
        const x = carry(this.pos.element(at)).toVar();
        const old = carry(this.prev.element(at));
        // Rebase history once per frame, before all of this frame's substeps use the new duration.
        this.prev.element(at).assign(x.sub(x.sub(old).mul(u.historyScale)));
        this.pos.element(at).assign(x);
        const bend = float(i)
          .sub(0.5)
          .div(this.points - 1)
          .mul(FUR.groom.bend)
          .add(FUR.groom.rootLean);
        baseline.addAssign(normalize(this.toWorld(n.add(lay.mul(bend)), 0)).mul(segment));
        this.rest.element(at).assign(baseline);
      });
      const h = u.step;
      const gravity = vec3(0, -SIM.gravity, 0);
      const parent = root.toVar();
      const target = root.toVar();
      Loop({ start: 0, end: u.steps, type: 'int' }, () => {
        parent.assign(root);
        target.assign(root);
        Loop({ start: 1, end: this.points, type: 'uint' }, ({ i }) => {
          const at = base.add(i);
          const x = this.pos.element(at).toVar();
          const velocity = x.sub(this.prev.element(at)).div(h);
          const along = float(i)
            .sub(1)
            .div(this.points - 2);
          const k = mix(SIM.springRoot, SIM.springTip, along);
          const bend = float(i)
            .sub(0.5)
            .div(this.points - 1)
            .mul(FUR.groom.bend)
            .add(FUR.groom.rootLean);
          const restDir = normalize(this.toWorld(n.add(lean.mul(bend)), 0));
          target.addAssign(restDir.mul(segment));
          const accel = gravity
            .add(target.sub(x).mul(k))
            .sub(velocity.mul(sqrt(k).mul(2 * SIM.dampingRatio)))
            .add(brushVel.sub(velocity).mul(comb.mul(SIM.brushDrag)));
          const move = velocity.mul(h).add(accel.mul(h).mul(h));
          const moved = length(move);
          const next = x.add(move.mul(min(moved, segment.mul(SIM.maxStep)).div(max(moved, 1e-6)))).toVar();
          // Bend resistance couples the segments to the groomed curve, preventing freely flapping tails.
          const edge = next.sub(parent);
          const dir = select(length(edge).greaterThan(1e-6), edge.div(max(length(edge), 1e-6)), restDir);
          const resistance = clamp(h.mul(float(1).sub(along.mul(0.6))).mul(16), 0, 0.2);
          next.assign(parent.add(normalize(mix(dir, restDir, resistance)).mul(segment)));
          // Project after the length constraint so it cannot put a point back inside a cube face.
          const local = u.anchorInv.mul(vec4(next, 1)).xyz;
          const r = max(length(local), 1e-6);
          const outward = local.div(r);
          const surface = baseR(outward).add(SIM.skin);
          If(r.lessThan(surface), () => next.assign(this.toWorld(outward.mul(surface), 1)));
          this.prev.element(at).assign(x);
          this.pos.element(at).assign(next);
          parent.assign(next);
        });
      });
      // Compute differences once per guide, instead of fetching both moving and resting curves for every
      // vertex of every follower. Both roots coincide, so their displacement above is exactly zero.
      Loop({ start: 1, end: this.points, type: 'uint' }, ({ i }) => {
        const at = base.add(i);
        this.displacement.element(at).assign(this.pos.element(at).sub(this.rest.element(at)));
      });
    })();
  }
}
