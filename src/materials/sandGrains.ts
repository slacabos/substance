import { DoubleSide, InstancedBufferGeometry, Mesh, PlaneGeometry, Vector4 } from 'three';
import {
  MeshBasicNodeMaterial,
  type ComputeNode,
  type Node,
  type StorageBufferNode,
  type UniformArrayNode,
  type WebGPURenderer,
} from 'three/webgpu';
import {
  atomicAdd,
  cameraProjectionMatrix,
  cross,
  dot,
  exp,
  float,
  Fn,
  hash,
  If,
  instancedArray,
  instanceIndex,
  int,
  length,
  max,
  min,
  modelViewMatrix,
  normalize,
  positionGeometry,
  Return,
  screenSize,
  select,
  smoothstep,
  storage,
  uint,
  uniform,
  uniformArray,
  varyingProperty,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { LOOKUP_SIZE } from '../core/oceanLookup';
import { cubeCell } from '../nodes/oceanSurface';
import { baseR } from '../nodes/shape';
import { LANDING, type SandField } from './sandField';

type Vec4Buffer = StorageBufferNode<'vec4'>;

/** Throws waiting for the next update, and the most grains one throws. */
const THROWS = 4;
export const PER_THROW = 32;
/** Sand one grain carries, orb radii cubed: a flick's worth of grains is about a tap's crater of sand. */
const GRAIN_VOLUME = 5e-7;
/** Radial gravity on grains and air drag, as the splashes (core/oceanFluid.ts). */
const GRAVITY = 1.3;
const DRAG = 0.3;
/** Seconds a grain may fly before it comes down wherever it is. */
const MOST_LIFE = 4;
/** Seconds of motion a grain's streak shows, and the longest it gets for its width: a blur, not a rice grain. */
const STREAK_TIME = 0.012;
const MOST_STRETCH = 1.8;

/**
 * Grains of sand thrown by a flick: each takes its sand from the bed under it as it leaves and gives it back where
 * it lands (atomically, through the field's `landings`), so the orb keeps every grain. They fly alone under
 * radial gravity: thrown grains are far apart, so they're drawn as small streaks rather than simulated as a
 * granular solid; once down they're part of the sand field again, which handles piles.
 */
export class SandGrains {
  readonly mesh: Mesh;
  private readonly renderer: WebGPURenderer;
  private readonly dt = uniform(0);
  private readonly frame = uniform(0);
  private readonly throws = Array.from({ length: THROWS }, () => new Vector4(0, 0, 1, 0));
  private readonly throwMotion = Array.from({ length: THROWS }, () => new Vector4());
  private pending = 0;
  private readonly throwNode: ComputeNode;
  private readonly stepNode: ComputeNode;
  private readonly resetNode: ComputeNode;

  constructor(renderer: WebGPURenderer, coarse: boolean, field: SandField) {
    this.renderer = renderer;
    const n = coarse ? 4096 : 16384;
    // position and life left (0 when not flying); velocity and size
    const pos = instancedArray(n, 'vec4') as Vec4Buffer;
    const vel = instancedArray(n, 'vec4') as Vec4Buffer;
    const head = instancedArray(1, 'uint').toAtomic();
    const throws = uniformArray(this.throws, 'vec4') as UniformArrayNode<'vec4'>;
    const motion = uniformArray(this.throwMotion, 'vec4') as UniformArrayNode<'vec4'>;
    const { bed } = field.renderBuffers();
    const grain = int(Math.round(GRAIN_VOLUME * LANDING));
    const vertexUnder = (d: Node<'vec3'>) => field.lookup.element(cubeCell(d, LOOKUP_SIZE));

    this.throwNode = Fn(() => {
      const e = instanceIndex.div(PER_THROW);
      const k = instanceIndex.mod(PER_THROW);
      const t = throws.element(e);
      const m = motion.element(e);
      // the throw's strength sets how many of its slots fly
      If(t.w.lessThanEqual(0).or(float(k).greaterThanEqual(t.w.mul(PER_THROW))), () => {
        Return();
      });
      const seed = uint(this.frame)
        .mul(THROWS * PER_THROW)
        .add(instanceIndex)
        .mul(8);
      const r = (j: number) => hash(seed.add(j));
      const d = normalize(t.xyz);
      const side = normalize(cross(d, m.xyz.add(d.mul(1e-4))));
      const spot = normalize(
        d.add(side.mul(r(0).sub(0.5).mul(0.06))).add(m.xyz.mul(r(1).sub(0.5).mul(0.04))),
      );
      const v = vertexUnder(spot);
      const slot = atomicAdd(head.element(0), uint(1)).mod(n);
      // off the surface, along the flick and up, scattered
      const speed = m.w.mul(r(2).mul(0.6).add(0.4));
      const velocity = m.xyz
        .mul(speed)
        .add(spot.mul(speed.mul(r(3).mul(0.6).add(0.3))))
        .add(side.mul(r(4).sub(0.5).mul(speed).mul(0.5)));
      const start = spot.mul(baseR(spot).add(bed.element(v).x).add(0.004));
      pos.element(slot).assign(vec4(start, MOST_LIFE));
      vel.element(slot).assign(vec4(velocity, r(5).mul(0.0025).add(0.0012)));
      atomicAdd(field.landings.element(v), grain.negate());
    })().compute(THROWS * PER_THROW);

    // fly; come down onto the sand (or wherever they are, once their time is up) and give it back
    this.stepNode = Fn(() => {
      const i = instanceIndex;
      const p = pos.element(i);
      If(p.w.lessThanEqual(0), () => {
        Return();
      });
      const g = vel.element(i);
      const d = normalize(p.xyz);
      const v = g.xyz.sub(d.mul(this.dt.mul(GRAVITY))).mul(exp(this.dt.mul(-DRAG)));
      const next = p.xyz.add(v.mul(this.dt));
      const under = vertexUnder(normalize(next));
      const ground = baseR(normalize(next)).add(bed.element(under).x);
      const down = length(next).lessThan(ground).and(dot(v, d).lessThan(0));
      const life = p.w.sub(this.dt);
      If(down.or(life.lessThanEqual(0)), () => {
        atomicAdd(field.landings.element(under), grain);
        p.assign(vec4(next, 0));
        Return();
      });
      p.assign(vec4(next, life));
      g.assign(vec4(v, g.w));
    })().compute(n);

    this.resetNode = Fn(() => {
      // grains still in the air go back where they are, so none is lost
      const p = pos.element(instanceIndex);
      If(p.w.greaterThan(0), () => {
        atomicAdd(field.landings.element(vertexUnder(normalize(p.xyz))), grain);
      });
      p.assign(vec4(0, 0, 1, 0));
    })().compute(n);

    // Specks of sand: round dots, smeared a little along their motion when fast (never longer than MOST_STRETCH
    // times their width). Grains far finer than a pixel are drawn a pixel wide but fainter, as much as they'd cover,
    // so a spray of them reads as a fine dust rather than a shower of rice.
    const posView = storage(pos.value, 'vec4', n).toReadOnly() as Vec4Buffer;
    const velView = storage(vel.value, 'vec4', n).toReadOnly() as Vec4Buffer;
    const corner = varyingProperty('vec2', 'vGrainCorner');
    // how much longer than wide the speck is, and how much of its pixel it covers
    const shape = varyingProperty('vec2', 'vGrainShape');
    const tint = varyingProperty('vec3', 'vGrainTint');
    const material = new MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: DoubleSide });
    material.vertexNode = Fn(() => {
      const i = instanceIndex;
      const p = posView.element(i);
      const g = velView.element(i);
      const at = modelViewMatrix.mul(vec4(p.xyz, 1));
      const flow = modelViewMatrix.mul(vec4(g.xyz, 0)).xy;
      const speed = length(flow);
      const along = select(speed.greaterThan(1e-5), flow.div(max(speed, 1e-5)), vec2(1, 0));
      const across = vec2(along.y.negate(), along.x);
      // a pixel, in view units at the grain's depth
      const clip = cameraProjectionMatrix.mul(at);
      const pixel = at.z
        .negate()
        .mul(2)
        .div(
          cameraProjectionMatrix
            .mul(vec4(0, 1, at.z, 0))
            .y.max(1e-6)
            .mul(screenSize.y),
        );
      const flying = select(p.w.greaterThan(0), float(1), float(0));
      const radius = max(g.w, pixel.mul(0.75));
      const reach = min(radius.add(speed.mul(STREAK_TIME)), radius.mul(MOST_STRETCH));
      shape.assign(vec2(reach.div(radius), min(float(1), g.w.div(radius)).pow(2)));
      corner.assign(positionGeometry.xy);
      const r = hash(i.add(n * 3));
      tint.assign(
        select(
          r.lessThan(0.1),
          vec3(0.3, 0.27, 0.24),
          select(
            r.lessThan(0.16),
            vec3(0.86, 0.8, 0.68),
            vec3(0.72, 0.6, 0.42).mul(
              hash(i.add(n * 5))
                .mul(0.35)
                .add(0.8),
            ),
          ),
        ),
      );
      const offset = along
        .mul(positionGeometry.x.mul(reach))
        .add(across.mul(positionGeometry.y.mul(radius)))
        .mul(flying);
      return clip.add(cameraProjectionMatrix.mul(vec4(offset, 0, 0)));
    })();
    material.colorNode = Fn(() => {
      // a capsule: distance from the segment along its middle, in radii
      const stretch = shape.x;
      const x = max(corner.x.abs().mul(stretch).sub(stretch.sub(1)), 0);
      const d = length(vec2(x, corner.y));
      const alpha = float(1)
        .sub(smoothstep(0.55, 1, d))
        .mul(shape.y);
      return vec4(tint, alpha.mul(0.95));
    })();
    const quad = new PlaneGeometry(2, 2);
    const geometry = new InstancedBufferGeometry();
    geometry.setIndex(quad.getIndex());
    geometry.setAttribute('position', quad.getAttribute('position'));
    geometry.instanceCount = n;
    this.mesh = new Mesh(geometry, material);
    this.mesh.frustumCulled = false;
  }

  /**
   * A flick at unit direction d, along unit tangent m at `speed` (orb radii per second): `share` (0 to 1) of a
   * throw's grains fly. Returns how many.
   */
  throw(d: readonly number[], m: readonly number[], speed: number, share: number): number {
    const slot = this.pending++ % THROWS;
    const count = Math.min(1, share);
    this.throws[slot]!.set(d[0]!, d[1]!, d[2]!, count);
    this.throwMotion[slot]!.set(m[0]!, m[1]!, m[2]!, speed);
    return Math.ceil(count * PER_THROW);
  }

  reset(): void {
    this.renderer.compute(this.resetNode);
  }

  update(dt: number): void {
    if (dt <= 0) return;
    this.dt.value = Math.min(dt, 1 / 30);
    this.frame.value = (this.frame.value + 1) % 4096;
    const passes = this.pending > 0 ? [this.throwNode, this.stepNode] : [this.stepNode];
    this.renderer.compute(passes);
    for (const t of this.throws) t.w = 0;
    this.pending = 0;
  }
}
