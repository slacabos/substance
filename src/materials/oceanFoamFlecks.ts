import { CircleGeometry, DoubleSide, InstancedBufferGeometry, Mesh, Vector3, Vector4 } from 'three';
import {
  MeshBasicNodeMaterial,
  type ComputeNode,
  type StorageBufferNode,
  type UniformArrayNode,
  type WebGPURenderer,
} from 'three/webgpu';
import {
  clamp,
  cross,
  dot,
  float,
  Fn,
  If,
  instancedArray,
  instanceIndex,
  length,
  min,
  normalize,
  positionGeometry,
  select,
  smoothstep,
  storage,
  uniform,
  uniformArray,
  vec3,
  vec4,
} from 'three/tsl';
import { OCEAN_DEFAULTS } from '../core/oceanTuning';
import { oceanWeatherAt, type DeformUniforms } from '../nodes/deform';
import { sampleSurface, type SurfaceBuffers } from '../nodes/oceanSurface';
import { oceanWaveVertex, type OceanWaveFields } from '../nodes/oceanWaves';
import { baseR } from '../nodes/shape';

/** What flecks ride on: the simulated surface, and the FFT waves the water shows. */
export interface FleckSurface {
  buffers: SurfaceBuffers;
  waves: OceanWaveFields;
  uniforms: DeformUniforms;
  /** The ocean mesh's vertex spacing: flecks follow the waves the mesh can show. */
  spacing: number;
}

const WIND = new Vector3(0.35, 0.9, 0.25).normalize();
const SIDE = new Vector3(1, 0, 0);
/** Flecks that can start in one frame. */
const SPAWNS = 24;

type Vec4Buffer = StorageBufferNode<'vec4'>;

/**
 * Lagrangian whitewater flecks advected over the globe by the simulated current, on the GPU. The CPU only
 * decides where new ones start; each frame a compute pass carries every fleck along the surface and fades it,
 * and the vertex shader lays it on the waves.
 */
export class OceanFoamFlecks {
  readonly mesh: Mesh;
  private readonly material: MeshBasicNodeMaterial;
  private readonly renderer: WebGPURenderer;
  private readonly count: number;
  private readonly spawns = Array.from({ length: SPAWNS }, () => new Vector4(0, 0, 1, 0));
  private readonly spawnShapes = Array.from({ length: SPAWNS }, () => new Vector4());
  private spawnCount = 0;
  private readonly dt = uniform(0);
  private readonly stepNode: ComputeNode;
  private readonly resetNode: ComputeNode;
  private readonly tangent = new Vector3();
  private readonly side = new Vector3();
  private head = 0;
  private seed = 28734621;

  constructor(renderer: WebGPURenderer, coarse: boolean, surface: FleckSurface) {
    this.renderer = renderer;
    this.count = coarse ? 240 : 900;
    const n = this.count;
    // Direction and remaining life. Then two per fleck (sharing a buffer, to stay within eight storage buffers):
    // duration, length, width and offshore share (1 out at sea, 0 at the bank); the direction it streams in,
    // and the simulated height under it.
    const fleck = instancedArray(n, 'vec4') as Vec4Buffer;
    const detail = instancedArray(n * 2, 'vec4') as Vec4Buffer;
    const spawns = uniformArray(this.spawns, 'vec4') as UniformArrayNode<'vec4'>;
    const spawnShapes = uniformArray(this.spawnShapes, 'vec4') as UniformArrayNode<'vec4'>;

    this.stepNode = Fn(() => {
      const i = instanceIndex;
      const state = fleck.element(i);
      const form = detail.element(i.mul(2));
      for (let s = 0; s < SPAWNS; s++) {
        const spawn = spawns.element(s);
        const spawnShape = spawnShapes.element(s);
        If(spawn.w.greaterThan(0).and(spawnShape.w.equal(float(i))), () => {
          state.assign(spawn);
          form.assign(vec4(spawnShape.xyz, 1));
        });
      }
      If(state.w.greaterThan(0), () => {
        const d = normalize(state.xyz);
        const here = sampleSurface(d, surface.buffers);
        const flow = here.flow.sub(d.mul(dot(here.flow, d)));
        const moved = normalize(d.add(flow.mul(this.dt)));
        state.assign(vec4(moved, select(here.shore.lessThanEqual(0), float(0), state.w.sub(this.dt))));
        const wind = vec3(WIND.x, WIND.y, WIND.z);
        const tangent = select(
          length(flow).greaterThan(1e-3),
          normalize(flow),
          normalize(wind.sub(moved.mul(dot(wind, moved)))),
        );
        detail.element(i.mul(2).add(1)).assign(vec4(tangent, here.height));
        form.w.assign(smoothstep(0, 0.1, here.shore));
      });
    })().compute(n);
    this.resetNode = Fn(() => {
      fleck.element(instanceIndex).assign(vec4(0, 0, 1, 0));
    })().compute(n);

    const fleckView = storage(fleck.value, 'vec4', n).toReadOnly() as Vec4Buffer;
    const detailView = storage(detail.value, 'vec4', n * 2).toReadOnly() as Vec4Buffer;
    this.material = new MeshBasicNodeMaterial({
      color: '#e0f4f4',
      transparent: true,
      opacity: 0.66,
      depthWrite: false,
      side: DoubleSide,
    });
    this.material.positionNode = Fn(() => {
      const state = fleckView.element(instanceIndex);
      const form = detailView.element(instanceIndex.mul(2));
      const streak = detailView.element(instanceIndex.mul(2).add(1));
      const d = normalize(state.xyz);
      const fade = clamp(min(state.w.div(0.45), form.x.sub(state.w).div(0.15)), 0, 1).mul(
        select(state.w.greaterThan(0), float(1), float(0)),
      );
      const side = normalize(cross(d, streak.xyz));
      // the same swell the ocean's vertices get, damped over the shallows
      const offshore = form.w;
      const weather = oceanWeatherAt(d, surface.uniforms);
      const wave = oceanWaveVertex(d, surface.waves, surface.uniforms, weather, surface.spacing);
      const direction = normalize(d.add(wave.offset.mul(offshore)));
      const lift = wave.height.mul(offshore.mul(0.82).add(0.18)).add(streak.w).add(0.004);
      const centre = direction.mul(baseR(direction).add(lift));
      return centre
        .add(streak.xyz.mul(positionGeometry.x.mul(form.y).mul(fade)))
        .add(side.mul(positionGeometry.y.mul(form.z).mul(fade)));
    })();
    const circle = new CircleGeometry(0.5, 6);
    const geometry = new InstancedBufferGeometry();
    geometry.setIndex(circle.getIndex());
    geometry.setAttribute('position', circle.getAttribute('position'));
    geometry.instanceCount = n;
    this.mesh = new Mesh(geometry, this.material);
    this.mesh.frustumCulled = false;
  }

  setIntensity(foam: number): void {
    this.material.opacity = Math.min(1, (0.66 * foam) / OCEAN_DEFAULTS.foam);
  }

  emit(dir: Vector3, flow: Vector3 | null, strength: number, count: number): void {
    this.tangent.copy(flow ?? WIND).addScaledVector(dir, -(flow ?? WIND).dot(dir));
    if (this.tangent.lengthSq() < 1e-5) this.tangent.copy(SIDE).addScaledVector(dir, -SIDE.dot(dir));
    this.tangent.normalize();
    this.side.crossVectors(dir, this.tangent).normalize();
    for (let i = 0; i < count && this.spawnCount < SPAWNS; i++) {
      const at = new Vector3()
        .copy(dir)
        .addScaledVector(this.tangent, (this.random() - 0.5) * 0.1 * strength)
        .addScaledVector(this.side, (this.random() - 0.5) * 0.11 * strength)
        .normalize();
      const duration = 1.1 + this.random() * 1.4;
      const length = (0.008 + this.random() * 0.014) * (0.5 + strength * 0.5);
      const width = 0.003 + this.random() * 0.005;
      this.spawns[this.spawnCount]!.set(at.x, at.y, at.z, duration);
      this.spawnShapes[this.spawnCount]!.set(duration, length, width, this.head++ % this.count);
      this.spawnCount++;
    }
  }

  clear(): void {
    this.spawnCount = 0;
    for (const spawn of this.spawns) spawn.w = 0;
    this.renderer.compute(this.resetNode);
  }

  update(dt: number): void {
    this.dt.value = dt;
    this.renderer.compute(this.stepNode);
    for (const spawn of this.spawns) spawn.w = 0;
    this.spawnCount = 0;
  }

  private random(): number {
    this.seed ^= this.seed << 13;
    this.seed ^= this.seed >>> 17;
    this.seed ^= this.seed << 5;
    return (this.seed >>> 0) / 0x100000000;
  }
}
