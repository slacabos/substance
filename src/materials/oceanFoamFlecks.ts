import {
  DoubleSide,
  CircleGeometry,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  InstancedMesh,
  Matrix4,
  Vector3,
} from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { attribute, Fn, normalize, positionLocal } from 'three/tsl';
import { OCEAN_DEFAULTS } from '../core/oceanTuning';
import type { OceanWeather } from '../core/oceanWeather';
import type { DeformUniforms } from '../nodes/deform';
import { oceanWaveVertex, type OceanWaveFields } from '../nodes/oceanWaves';
import { baseR } from '../nodes/shape';
import { morphUniform } from '../scene/shape';
import type { OceanSim } from './oceanSim';

/** The ocean surface's FFT waves, so flecks ride the same swell the water shows. */
export interface FleckWaves {
  waves: OceanWaveFields;
  uniforms: DeformUniforms;
  /** The ocean mesh's vertex spacing: flecks follow the waves the mesh can show. */
  spacing: number;
}

interface Fleck {
  dir: Vector3;
  life: number;
  duration: number;
  length: number;
  width: number;
}

const WIND = new Vector3(0.35, 0.9, 0.25).normalize();
const SIDE = new Vector3(1, 0, 0);

/** Lagrangian whitewater flecks advected over the globe by the simulated tangential current. */
export class OceanFoamFlecks {
  readonly mesh: InstancedMesh;
  private readonly fleckMaterial: MeshBasicNodeMaterial;
  /** Per fleck: local storm strength, and how much of the swell reaches it (less over the shallows). */
  private readonly waveState: InstancedBufferAttribute;
  private readonly flecks: Fleck[];
  private readonly basis = new Matrix4();
  private readonly matrix = new Matrix4();
  private readonly scale = new Vector3();
  private readonly flow = new Vector3();
  private readonly tangent = new Vector3();
  private readonly side = new Vector3();
  private readonly surface = new Vector3();
  private head = 0;
  private seed = 28734621;

  constructor(coarse: boolean, surface?: FleckWaves) {
    const count = coarse ? 240 : 900;
    this.fleckMaterial = new MeshBasicNodeMaterial({
      color: '#e0f4f4',
      transparent: true,
      opacity: 0.66,
      depthWrite: false,
      side: DoubleSide,
    });
    const geometry = new CircleGeometry(0.5, 6);
    this.waveState = new InstancedBufferAttribute(new Float32Array(count * 2), 2).setUsage(DynamicDrawUsage);
    geometry.setAttribute('fleckWave', this.waveState);
    if (surface) {
      // The CPU places each fleck on the undisturbed orb plus the simulated height; the swell is added here,
      // the same way the ocean's vertices get it.
      this.fleckMaterial.positionNode = Fn(() => {
        const p = normalize(positionLocal);
        const state = attribute('fleckWave', 'vec2');
        const wave = oceanWaveVertex(p, surface.waves, surface.uniforms, state.x, surface.spacing);
        const damping = state.y.mul(0.82).add(0.18);
        const direction = normalize(p.add(wave.offset.mul(state.y)));
        const moved = direction.mul(baseR(direction).add(wave.height.mul(damping))).sub(p.mul(baseR(p)));
        return positionLocal.add(moved);
      })();
    }
    this.mesh = new InstancedMesh(geometry, this.fleckMaterial, count);
    this.mesh.frustumCulled = false;
    this.mesh.instanceMatrix.setUsage(DynamicDrawUsage);
    this.flecks = Array.from({ length: count }, () => ({
      dir: new Vector3(0, 0, 1),
      life: 0,
      duration: 0,
      length: 0,
      width: 0,
    }));
    this.clear();
  }

  setIntensity(foam: number): void {
    this.fleckMaterial.opacity = Math.min(1, (0.66 * foam) / OCEAN_DEFAULTS.foam);
  }

  emit(dir: Vector3, flow: Vector3 | null, strength: number, count: number): void {
    this.tangent.copy(flow ?? WIND).addScaledVector(dir, -(flow ?? WIND).dot(dir));
    if (this.tangent.lengthSq() < 1e-5) this.tangent.copy(SIDE).addScaledVector(dir, -SIDE.dot(dir));
    this.tangent.normalize();
    this.side.crossVectors(dir, this.tangent).normalize();
    for (let i = 0; i < count; i++) {
      const fleck = this.flecks[this.head++ % this.flecks.length]!;
      fleck.dir
        .copy(dir)
        .addScaledVector(this.tangent, (this.random() - 0.5) * 0.1 * strength)
        .addScaledVector(this.side, (this.random() - 0.5) * 0.11 * strength)
        .normalize();
      fleck.duration = 1.1 + this.random() * 1.4;
      fleck.life = fleck.duration;
      fleck.length = (0.008 + this.random() * 0.014) * (0.5 + strength * 0.5);
      fleck.width = 0.003 + this.random() * 0.005;
    }
  }

  clear(): void {
    for (const fleck of this.flecks) fleck.life = 0;
    this.update(0);
  }

  update(dt: number, sim?: OceanSim, weather?: OceanWeather): void {
    for (let i = 0; i < this.flecks.length; i++) {
      const fleck = this.flecks[i]!;
      if (fleck.life > 0 && sim && weather && dt > 0) {
        const state = sim.sampleAt(fleck.dir);
        if (state.shoreDistance <= 0) {
          fleck.life = 0;
          this.matrix.makeScale(0, 0, 0);
          this.mesh.setMatrixAt(i, this.matrix);
          continue;
        }
        this.flow.set(state.flowX, state.flowY, state.flowZ);
        this.flow.addScaledVector(fleck.dir, -this.flow.dot(fleck.dir));
        fleck.dir.addScaledVector(this.flow, dt).normalize();
        fleck.life -= dt;
        const t = Math.min(1, Math.max(0, state.shoreDistance / 0.1));
        this.waveState.setXY(i, weather.strengthAt(fleck.dir), t * t * (3 - 2 * t));
        const p = fleck.dir;
        const power = Math.abs(p.x) ** 10 + Math.abs(p.y) ** 10 + Math.abs(p.z) ** 10;
        const radius = 1 + (0.8 / power ** 0.1 - 1) * morphUniform.value;
        this.surface.copy(p).multiplyScalar(radius + state.height + 0.004);
        this.tangent.copy(this.flow).addScaledVector(fleck.dir, -this.flow.dot(fleck.dir));
        if (this.tangent.lengthSq() < 1e-5)
          this.tangent.copy(WIND).addScaledVector(fleck.dir, -WIND.dot(fleck.dir));
        this.tangent.normalize();
        this.side.crossVectors(fleck.dir, this.tangent).normalize();
        this.basis.makeBasis(this.tangent, this.side, fleck.dir);
        const fade = Math.min(1, fleck.life / 0.45, (fleck.duration - fleck.life) / 0.15);
        this.scale.set(fleck.length * Math.max(0, fade), fleck.width * Math.max(0, fade), 1);
        this.matrix.copy(this.basis).scale(this.scale).setPosition(this.surface);
      } else {
        this.scale.set(0, 0, 0);
        this.matrix.makeScale(0, 0, 0);
      }
      this.mesh.setMatrixAt(i, this.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    this.waveState.needsUpdate = true;
  }

  private random(): number {
    this.seed ^= this.seed << 13;
    this.seed ^= this.seed >>> 17;
    this.seed ^= this.seed << 5;
    return (this.seed >>> 0) / 0x100000000;
  }
}
