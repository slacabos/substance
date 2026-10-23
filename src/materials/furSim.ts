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
  sin,
  smoothstep,
  sqrt,
  storage,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import { FUR } from '../config';
import { baseN, baseR } from '../nodes/shape';

/** Tuning for the strand simulation. */
const SIM = {
  /** Fixed physics rate; each frame runs however many steps fit its time (at most maxSteps). */
  rate: 240,
  maxSteps: 8,
  gravity: 7,
  /**
   * Each point is a damped spring toward the strand standing straight along its rest direction: stiff at the
   * root, soft at the tip (per second squared). Tip sag under gravity is about gravity / springTip.
   */
  springRoot: 1200,
  springTip: 80,
  /** Damping ratio: below 1 the hair bounces a little before it settles. */
  dampingRatio: 0.3,
  /** Share of the body's own motion that turns into hair inertia; the rest carries the hair along. */
  inertia: 0.5,
  /** Cap on how far a point moves per step, in segment lengths, so violent flicks stay well behaved. */
  maxStep: 0.5,
  /** How quickly hair under a moving brush picks up the brush's velocity, per second. */
  brushDrag: 25,
  /** How fast a combed parting relaxes, per second. */
  groomRelax: 0.12,
  /** How strongly a stroke lays the hair over (tangent lean per unit stroke strength). */
  combLean: 2.2,
  /** Keeps points just off the body. */
  skin: 0.008,
} as const;

/**
 * Per-strand hair simulation on the GPU. Each strand is a chain of points from a root pinned to the
 * spinning body, stepped at a fixed rate: Verlet integration with gravity and inertia, a damped spring pulling
 * each point toward the strand's rest pose (standing along the surface normal, leaned over by combing and
 * pressing), collision with the body, then follow-the-leader length constraints. Points are in world space.
 */
export class FurSim {
  readonly count: number;
  /** Points per strand, root included. */
  readonly points: number;
  readonly uniforms = {
    anchor: uniform(new Matrix4()),
    anchorInv: uniform(new Matrix4()),
    /** Last frame's anchorInv, to carry the hair along with the body's motion. */
    anchorPrevInv: uniform(new Matrix4()),
    scale: uniform(1),
    /** Frame time, and the fixed steps it's split into. */
    dt: uniform(1 / 60),
    steps: uniform(4, 'int'),
    step: uniform(1 / SIM.rate),
    time: uniform(0),
    len: uniform(FUR.length),
    /** Object-space unit direction of the brush contact. */
    brushDir: uniform(new Vector3(0, 0, 1)),
    /** Object-space tangent lean the brush combs toward. */
    brushStroke: uniform(new Vector3()),
    /** Object-space velocity of the brush along the surface, units per second. */
    brushVel: uniform(new Vector3()),
    /** 1 while a stroke is moving, decays to 0 when it stops. */
    brushAmount: uniform(0),
    /** Object-space press direction and how hard. */
    pressDir: uniform(new Vector3(0, 0, 1)),
    press: uniform(0),
  };

  private readonly renderer: WebGPURenderer;
  private readonly roots: StorageBufferNode<'vec3'>;
  private readonly rand: StorageBufferNode<'vec3'>;
  private readonly pos: StorageBufferNode<'vec3'>;
  private readonly prev: StorageBufferNode<'vec3'>;
  private readonly groom: StorageBufferNode<'vec3'>;
  private readonly stepNode: ComputeNode;
  private readonly resetNode: ComputeNode;

  constructor(renderer: WebGPURenderer, roots: Float32Array, rand: Float32Array, points: number) {
    this.renderer = renderer;
    this.count = roots.length / 3;
    this.points = points;
    this.roots = instancedArray(roots, 'vec3');
    this.rand = instancedArray(rand, 'vec3');
    this.pos = instancedArray(this.count * points, 'vec3');
    this.prev = instancedArray(this.count * points, 'vec3');
    this.groom = instancedArray(this.count, 'vec3');
    this.stepNode = this.createStep().compute(this.count);
    this.resetNode = this.createReset().compute(this.count);
  }

  /** Read-only views of the root directions, random numbers and points, for the strand vertex shader. */
  renderBuffers() {
    const view = (node: StorageBufferNode<'vec3'>) =>
      storage(node.value, 'vec3', node.bufferCount).toReadOnly() as StorageBufferNode<'vec3'>;
    return { roots: view(this.roots), rand: view(this.rand), pos: view(this.pos) };
  }

  /** Puts every strand back at rest along its surface normal. */
  reset(anchor: Object3D): void {
    this.setAnchor(anchor);
    this.uniforms.anchorPrevInv.value.copy(this.uniforms.anchorInv.value);
    this.renderer.compute(this.resetNode);
  }

  update(anchor: Object3D, time: number, dt: number): void {
    this.setAnchor(anchor);
    this.uniforms.time.value = time;
    const steps = Math.min(SIM.maxSteps, Math.max(1, Math.round(dt * SIM.rate)));
    this.uniforms.dt.value = dt;
    this.uniforms.steps.value = steps;
    this.uniforms.step.value = dt / steps;
    this.renderer.compute(this.stepNode);
  }

  private setAnchor(anchor: Object3D): void {
    this.uniforms.anchorPrevInv.value.copy(this.uniforms.anchorInv.value);
    anchor.updateMatrixWorld();
    this.uniforms.anchor.value.copy(anchor.matrixWorld);
    this.uniforms.anchorInv.value.copy(anchor.matrixWorld).invert();
    this.uniforms.scale.value = anchor.scale.x;
  }

  /** The strand's root on the body (object space), its surface normal and its world segment length. */
  private strandBase() {
    const u = this.uniforms;
    const p = this.roots.element(instanceIndex);
    const q = p.mul(baseR(p));
    const n = baseN(p, q);
    const segment = u.len
      .mul(this.rand.element(instanceIndex).x.mul(0.45).add(0.55))
      .mul(u.scale)
      .div(this.points - 1);
    return { p, q, n, segment };
  }

  private toWorld(v: Node<'vec3'>, w: number): Node<'vec3'> {
    return this.uniforms.anchor.mul(vec4(v, w)).xyz;
  }

  private createReset() {
    return Fn(() => {
      If(instanceIndex.greaterThanEqual(this.count), () => {
        Return();
      });
      const { q, n, segment } = this.strandBase();
      const root = this.toWorld(q, 1);
      const dir = normalize(this.toWorld(n, 0));
      const base = instanceIndex.mul(this.points);
      Loop({ start: 0, end: this.points, type: 'uint' }, ({ i }) => {
        const x = root.add(dir.mul(segment).mul(float(i)));
        this.pos.element(base.add(i)).assign(x);
        this.prev.element(base.add(i)).assign(x);
      });
      this.groom.element(instanceIndex).assign(vec3(0));
    })();
  }

  private createStep() {
    const u = this.uniforms;
    return Fn(() => {
      If(instanceIndex.greaterThanEqual(this.count), () => {
        Return();
      });
      const { p, q, n, segment } = this.strandBase();

      // Combing: while a stroke passes over, the strand's groom turns toward the stroke; it relaxes slowly.
      const groom = this.groom.element(instanceIndex);
      const bs = acos(clamp(dot(p, u.brushDir), -1, 1));
      const comb = exp(bs.mul(bs).div(-0.05)).mul(u.brushAmount);
      const combed = mix(groom, u.brushStroke.mul(SIM.combLean), clamp(comb.mul(u.dt).mul(10), 0, 1)).mul(
        exp(u.dt.mul(-SIM.groomRelax)),
      );
      groom.assign(combed.sub(n.mul(dot(combed, n))));

      // Pressing parts the fur around the contact; a little sway keeps it alive.
      const away = p.sub(u.pressDir.mul(dot(p, u.pressDir)));
      const ps = acos(clamp(dot(p, u.pressDir), -1, 1));
      const part = normalize(away)
        .mul(u.press.mul(1.3))
        .mul(exp(ps.mul(ps).div(-0.035)))
        .mul(smoothstep(0, 0.08, ps))
        .mul(select(length(away).greaterThan(1e-4), float(1), float(0)));
      const t = u.time;
      const sway = vec3(
        sin(t.mul(1.3).add(p.y.mul(5))),
        sin(t.mul(1.1).add(p.z.mul(4))),
        sin(t.mul(1.7).add(p.x.mul(4.5))),
      ).mul(0.04);
      const lean = groom.add(part).add(sway);
      const rest = normalize(this.toWorld(n.add(lean.sub(n.mul(dot(lean, n)))), 0));
      const brushVel = this.toWorld(u.brushVel, 0);

      const root = this.toWorld(q, 1);
      const base = instanceIndex.mul(this.points);
      this.pos.element(base).assign(root);
      this.prev.element(base).assign(root);

      // Carry the hair part of the way along with the body's motion since last frame (velocities included),
      // so only some of a spin turns into trailing; all of it would sweep the hair flat on a hard flick.
      const carry = (x: Node<'vec3'>) =>
        mix(x, this.toWorld(u.anchorPrevInv.mul(vec4(x, 1)).xyz, 1), 1 - SIM.inertia);
      Loop({ start: 1, end: this.points, type: 'uint' }, ({ i }) => {
        const at = base.add(i);
        this.pos.element(at).assign(carry(this.pos.element(at)));
        this.prev.element(at).assign(carry(this.prev.element(at)));
      });

      const h = u.step;
      const gravity = vec3(0, -SIM.gravity, 0);
      const parent = root.toVar();
      Loop({ start: 0, end: u.steps, type: 'int' }, () => {
        parent.assign(root);
        Loop({ start: 1, end: this.points, type: 'uint' }, ({ i }) => {
          const at = base.add(i);
          const x = this.pos.element(at).toVar();
          const velocity = x.sub(this.prev.element(at)).div(h);
          const along = float(i)
            .sub(1)
            .div(this.points - 2);
          const k = mix(SIM.springRoot, SIM.springTip, along);
          const target = root.add(rest.mul(segment).mul(float(i)));

          // gravity, a damped spring toward the rest pose, and the drag of a brush moving through the hair
          const accel = gravity
            .add(target.sub(x).mul(k))
            .sub(velocity.mul(sqrt(k).mul(2 * SIM.dampingRatio)))
            .add(brushVel.sub(velocity).mul(comb.mul(SIM.brushDrag)));
          const move = velocity.mul(h).add(accel.mul(h).mul(h));
          const moved = length(move);
          const capped = move.mul(min(moved, segment.mul(SIM.maxStep)).div(max(moved, 1e-6)));
          const next = x.add(capped).toVar();

          // stay outside the body
          const local = u.anchorInv.mul(vec4(next, 1)).xyz;
          const r = length(local);
          const dir = local.div(r);
          const surface = baseR(dir).add(SIM.skin);
          If(r.lessThan(surface), () => {
            next.assign(this.toWorld(dir.mul(surface), 1));
          });

          // follow the leader: exactly one segment from the parent
          next.assign(parent.add(normalize(next.sub(parent)).mul(segment)));
          this.prev.element(at).assign(x);
          this.pos.element(at).assign(next);
          parent.assign(next);
        });
      });
    })();
  }
}
