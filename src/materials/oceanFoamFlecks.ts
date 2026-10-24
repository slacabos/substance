import { CircleGeometry, DoubleSide, InstancedBufferGeometry, Mesh, Vector3 } from 'three';
import {
  MeshBasicNodeMaterial,
  type ComputeNode,
  type StorageBufferNode,
  type WebGPURenderer,
} from 'three/webgpu';
import {
  atomicAdd,
  clamp,
  cross,
  dot,
  float,
  Fn,
  hash,
  If,
  instancedArray,
  instanceIndex,
  length,
  min,
  normalize,
  positionGeometry,
  Return,
  select,
  smoothstep,
  storage,
  uint,
  uniform,
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
  /** How many vertices the surface has. */
  count: number;
}

const WIND = new Vector3(0.35, 0.9, 0.25).normalize();
/** Flecks started per second per unit of sphere area where the water is all whitewater, for 900 flecks. */
const EMISSION = 4000;

type Vec4Buffer = StorageBufferNode<'vec4'>;

/**
 * Lagrangian whitewater flecks advected over the globe by the simulated current, all on the GPU. Each frame one
 * pass over the surface's vertices starts flecks where the water is breaking (claiming the oldest slots of a
 * ring), another carries every fleck along the surface and fades it, and the vertex shader lays it on the
 * waves.
 */
export class OceanFoamFlecks {
  readonly mesh: Mesh;
  private readonly material: MeshBasicNodeMaterial;
  private readonly renderer: WebGPURenderer;
  private readonly count: number;
  private readonly dt = uniform(0);
  /** Frame number (wrapping), so each frame draws new random numbers. */
  private readonly frame = uniform(0);
  private readonly emitNode: ComputeNode;
  private readonly stepNode: ComputeNode;
  private readonly resetNode: ComputeNode;

  constructor(renderer: WebGPURenderer, coarse: boolean, surface: FleckSurface) {
    this.renderer = renderer;
    this.count = coarse ? 240 : 900;
    const n = this.count;
    // Direction and remaining life. Then two per fleck (sharing a buffer, to stay within eight storage buffers):
    // duration, length, width and offshore share (1 out at sea, 0 at the bank); the direction it streams in,
    // and the simulated height under it.
    const fleck = instancedArray(n, 'vec4') as Vec4Buffer;
    const detail = instancedArray(n * 2, 'vec4') as Vec4Buffer;
    const { buffers } = surface;

    // Where the water is breaking, by the same measure as spray sites (OceanSim): surf on a coast's breaking
    // band, aerated water elsewhere (a pressed dent is turbulent but holds no air).
    const head = instancedArray(1, 'uint').toAtomic();
    const budget = (EMISSION * n) / 900;
    this.emitNode = Fn(() => {
      const i = instanceIndex;
      const ground = buffers.ground.element(i);
      If(ground.x.lessThanEqual(0), () => {
        Return();
      });
      const here = buffers.surface.element(i);
      const vertex = buffers.vertex.element(i);
      const coastSite = ground.y.lessThan(0.09);
      const aerated = select(here.y.greaterThanEqual(0.18), here.y.mul(0.82).add(here.z.mul(0.3)), float(0));
      const score = select(coastSite, here.w, aerated);
      const threshold = select(coastSite, float(0.22), float(0.18));
      const excess = clamp(score.sub(threshold).div(float(1).sub(threshold)), 0, 1);
      const seed = uint(this.frame).mul(surface.count).add(i).mul(8);
      const random = (k: number) => hash(seed.add(k));
      If(random(0).lessThan(excess.mul(budget).mul(vertex.w).mul(this.dt)), () => {
        const slot = atomicAdd(head.element(0), uint(1)).mod(n);
        const jitter = vec3(random(1), random(2), random(3))
          .sub(0.5)
          .mul(surface.spacing * 1.2);
        const duration = random(4).mul(1.4).add(1.1);
        const strength = clamp(score, 0.25, 0.95);
        const streak = random(5).mul(0.014).add(0.008).mul(strength.mul(0.5).add(0.5));
        const width = random(6).mul(0.005).add(0.003);
        fleck.element(slot).assign(vec4(normalize(vertex.xyz.add(jitter)), duration));
        detail.element(slot.mul(2)).assign(vec4(duration, streak, width, 1));
      });
    })().compute(surface.count);

    this.stepNode = Fn(() => {
      const i = instanceIndex;
      const state = fleck.element(i);
      const form = detail.element(i.mul(2));
      If(state.w.greaterThan(0), () => {
        const d = normalize(state.xyz);
        const here = sampleSurface(d, buffers);
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

  clear(): void {
    this.renderer.compute(this.resetNode);
  }

  update(dt: number): void {
    this.dt.value = dt;
    this.frame.value = (this.frame.value + 1) % 4096;
    this.renderer.compute([this.emitNode, this.stepNode]);
  }
}
