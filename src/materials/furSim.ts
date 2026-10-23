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
  storage,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import { FUR } from '../config';
import { baseN, baseR } from '../nodes/shape';

/** Tuning for the strand simulation. Stiffness is how far a segment swings back to rest per substep. */
const SIM = {
  substeps: 3,
  gravity: 12,
  damping: 0.985,
  stiffnessRoot: 0.12,
  stiffnessTip: 0.01,
  /** How much each segment also lines up with the one before it. */
  bend: 0.15,
  /** Furthest the first segment leans from upright, and each joint bends, in radians. Stops curls and loops. */
  maxRootAngle: 0.96,
  maxJointAngle: 0.32,
  /** Share of the body's own motion the hair is carried along with each frame; the rest shows as trailing. */
  carry: 0.7,
  /** Cap on how far a point moves per substep, in segment lengths, so violent flicks don't curl strands. */
  maxStep: 0.9,
  /** Share of the brush's velocity passed to the hair under it. */
  brushPush: 0.8,
  /** How fast a combed parting relaxes, per second. */
  groomRelax: 0.12,
  /** How strongly a stroke lays the hair over (tangent lean per unit stroke strength). */
  combLean: 2.2,
  /** Keeps points just off the body. */
  skin: 0.008,
} as const;

/**
 * Per-strand hair simulation on the GPU. Each strand is a chain of points from a root pinned to the
 * spinning body: Verlet integration with gravity and inertia, a stiffness pull toward a rest direction (the
 * surface normal, leaned over by combing and pressing), collision with the body, then follow-the-leader length
 * constraints, which keep it stable however hard the object is flicked. Points are in world space.
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
    /** Frame time; the shader splits it into substeps. */
    dt: uniform(1 / 60),
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
    this.uniforms.dt.value = dt;
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
      ).mul(0.06);
      const lean = groom.add(part).add(sway);
      const rest = normalize(this.toWorld(n.add(lean.sub(n.mul(dot(lean, n)))), 0));

      // the brush drags the hair under it along with the stroke
      const push = this.toWorld(u.brushVel, 0).mul(comb.mul(SIM.brushPush));

      const root = this.toWorld(q, 1);
      const base = instanceIndex.mul(this.points);
      this.pos.element(base).assign(root);
      this.prev.element(base).assign(root);
      const parent = root.toVar();
      const h = u.dt.div(SIM.substeps);
      const gravity = vec3(0, -SIM.gravity, 0).mul(h).mul(h);

      // Carry the hair most of the way along with the body's motion since last frame (velocities included),
      // so only part of a spin turns into trailing; otherwise a hard flick would sweep it flat.
      const carry = (x: Node<'vec3'>) =>
        mix(x, this.toWorld(u.anchorPrevInv.mul(vec4(x, 1)).xyz, 1), SIM.carry);
      Loop({ start: 1, end: this.points, type: 'uint' }, ({ i }) => {
        const at = base.add(i);
        this.pos.element(at).assign(carry(this.pos.element(at)));
        this.prev.element(at).assign(carry(this.prev.element(at)));
      });

      // substeps run inside the one dispatch: separate dispatches per substep cost a frame on some GPUs
      const parentDir = vec3(0).toVar();
      Loop(SIM.substeps, () => {
        parent.assign(root);
        parentDir.assign(rest);
        Loop({ start: 1, end: this.points, type: 'uint' }, ({ i }) => {
          const x = this.pos.element(base.add(i)).toVar();
          const old = this.prev.element(base.add(i));
          const along = float(i)
            .sub(1)
            .div(this.points - 2);
          const stiffness = mix(SIM.stiffnessRoot, SIM.stiffnessTip, along);

          // inertia + gravity, then line up a little with the parent segment and swing part of the way back
          // toward the rest direction
          const velocity = x.sub(old).mul(SIM.damping);
          const speed = length(velocity);
          const capped = velocity.mul(min(speed, segment.mul(SIM.maxStep)).div(max(speed, 1e-6)));
          const predicted = x
            .add(capped)
            .add(gravity)
            .add(push.mul(h).mul(along.mul(0.6).add(0.4)));
          const smoothed = mix(normalize(predicted.sub(parent)), parentDir, SIM.bend);
          const swung = normalize(mix(normalize(smoothed), rest, stiffness));

          // never fold more than a set angle from the segment before (from upright, for the first one)
          const first = i.equal(1);
          const cosMax = select(first, float(Math.cos(SIM.maxRootAngle)), float(Math.cos(SIM.maxJointAngle)));
          const sinMax = select(first, float(Math.sin(SIM.maxRootAngle)), float(Math.sin(SIM.maxJointAngle)));
          const c = dot(swung, parentDir);
          const across = normalize(swung.sub(parentDir.mul(c)));
          const limited = select(c.lessThan(cosMax), parentDir.mul(cosMax).add(across.mul(sinMax)), swung);
          const next = parent.add(limited.mul(segment)).toVar();

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
          this.prev.element(base.add(i)).assign(x);
          this.pos.element(base.add(i)).assign(next);
          parentDir.assign(normalize(next.sub(parent)));
          parent.assign(next);
        });
      });
    })();
  }
}
