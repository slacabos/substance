import {
  Color,
  DynamicDrawUsage,
  InstancedMesh,
  Matrix4,
  MeshBasicMaterial,
  MeshPhysicalMaterial,
  Quaternion,
  SphereGeometry,
  Vector3,
} from 'three';
import { morphUniform } from '../scene/shape';
import type { OceanSim } from './oceanSim';

const UP = new Vector3(0, 1, 0);
const SIDE = new Vector3(1, 0, 0);

interface Drop {
  pos: Vector3;
  vel: Vector3;
  life: number;
  duration: number;
  size: number;
}

/** Droplets and fine mist emitted by breaking lips, returning to the orb under radial gravity. */
export class OceanSpray {
  readonly mesh: InstancedMesh;
  readonly mistMesh: InstancedMesh;
  private readonly drops: Drop[];
  private readonly mist: Drop[];
  private readonly matrix = new Matrix4();
  private readonly scale = new Vector3();
  private readonly rotation = new Quaternion();
  private readonly tangentA = new Vector3();
  private readonly tangentB = new Vector3();
  private readonly tmp = new Vector3();
  private dropHead = 0;
  private mistHead = 0;
  private seed = 123456789;

  constructor(coarse: boolean) {
    const dropCount = coarse ? 180 : 800;
    const mistCount = coarse ? 350 : 1600;
    const dropletMaterial = new MeshPhysicalMaterial({
      color: new Color('#c7eaf2'),
      metalness: 0,
      roughness: 0.18,
      transmission: 0.18,
      ior: 1.333,
      transparent: true,
      opacity: 0.83,
      depthWrite: false,
      envMapIntensity: 0.6,
    });
    const mistMaterial = new MeshBasicMaterial({
      color: '#d8f2f3',
      transparent: true,
      opacity: 0.38,
      depthWrite: false,
    });
    this.mesh = new InstancedMesh(new SphereGeometry(1, 5, 4), dropletMaterial, dropCount);
    this.mistMesh = new InstancedMesh(new SphereGeometry(1, 3, 2), mistMaterial, mistCount);
    this.mesh.frustumCulled = false;
    this.mistMesh.frustumCulled = false;
    this.mesh.instanceMatrix.setUsage(DynamicDrawUsage);
    this.mistMesh.instanceMatrix.setUsage(DynamicDrawUsage);
    const make = (): Drop => ({ pos: new Vector3(), vel: new Vector3(), life: 0, duration: 0, size: 0 });
    this.drops = Array.from({ length: dropCount }, make);
    this.mist = Array.from({ length: mistCount }, make);
    this.update(0);
  }

  splash(dir: Vector3, strength: number, count = 20): void {
    this.emitCrest(dir, null, strength, count);
  }

  emitCrest(
    dir: Vector3,
    travel: Vector3 | null,
    strength: number,
    count: number,
    surfaceHeight?: number,
  ): void {
    this.tangentA.crossVectors(dir, Math.abs(dir.y) < 0.9 ? UP : SIDE).normalize();
    this.tangentB.crossVectors(dir, this.tangentA).normalize();
    const origin = this.surfaceRadius(dir) + (surfaceHeight === undefined ? 0.035 : surfaceHeight + 0.012);
    for (let i = 0; i < count; i++) {
      const drop = this.drops[this.dropHead++ % this.drops.length]!;
      this.launch(drop, dir, travel, origin, strength, false);
    }
    for (let i = 0; i < count * 3; i++) {
      const drop = this.mist[this.mistHead++ % this.mist.length]!;
      this.launch(drop, dir, travel, origin, strength, true);
    }
  }

  clear(): void {
    for (const drop of this.drops) drop.life = 0;
    for (const drop of this.mist) drop.life = 0;
    this.update(0);
  }

  update(dt: number, sim?: OceanSim): void {
    this.updatePool(this.drops, this.mesh, dt, sim, false);
    this.updatePool(this.mist, this.mistMesh, dt, sim, true);
  }

  private launch(
    drop: Drop,
    dir: Vector3,
    travel: Vector3 | null,
    origin: number,
    strength: number,
    mist: boolean,
  ): void {
    const angle = this.random() * Math.PI * 2;
    const spread = Math.sqrt(this.random()) * (mist ? 0.065 : 0.09) * strength;
    const x = Math.cos(angle);
    const y = Math.sin(angle);
    drop.pos.copy(dir).multiplyScalar(origin + this.random() * 0.018);
    drop.pos.addScaledVector(this.tangentA, x * spread);
    drop.pos.addScaledVector(this.tangentB, y * spread);
    drop.vel.copy(dir).multiplyScalar((mist ? 0.22 : 0.33) * strength + this.random() * 0.15);
    if (travel && travel.lengthSq() > 1e-5)
      drop.vel.addScaledVector(travel, ((0.22 + this.random() * 0.24) * strength) / travel.length());
    drop.vel.addScaledVector(this.tangentA, x * (0.12 + this.random() * 0.16) * strength);
    drop.vel.addScaledVector(this.tangentB, y * (0.12 + this.random() * 0.16) * strength);
    drop.duration = mist ? 0.28 + this.random() * 0.55 : 0.5 + this.random() * 0.6;
    drop.life = drop.duration;
    drop.size = mist ? 0.002 + this.random() * 0.004 : 0.003 + Math.pow(this.random(), 3) * 0.007;
  }

  private updatePool(
    pool: Drop[],
    mesh: InstancedMesh,
    dt: number,
    sim: OceanSim | undefined,
    mist: boolean,
  ): void {
    for (let i = 0; i < pool.length; i++) {
      const drop = pool[i]!;
      if (drop.life > 0 && dt > 0) {
        this.tmp.copy(drop.pos).normalize();
        drop.vel.addScaledVector(this.tmp, -(mist ? 0.9 : 1.3) * dt);
        drop.vel.multiplyScalar(Math.exp(-(mist ? 1.3 : 0.45) * dt));
        drop.pos.addScaledVector(drop.vel, dt);
        drop.life -= dt;
        this.tmp.copy(drop.pos).normalize();
        const ground = sim?.groundHeight(this.tmp) ?? 0;
        if (drop.pos.length() < this.surfaceRadius(this.tmp) + Math.max(0.012, ground + 0.004)) {
          drop.life = 0;
          if (sim) sim.depositFoam(this.tmp, mist ? 0.004 : 0.012, mist ? 0.035 : 0.05);
        }
      }
      const fade = Math.min(1, Math.max(0, drop.life / (mist ? 0.22 : 0.15)));
      this.scale.setScalar(drop.life > 0 ? drop.size * fade : 0);
      this.matrix.compose(drop.pos, this.rotation, this.scale);
      mesh.setMatrixAt(i, this.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }

  private surfaceRadius(dir: Vector3): number {
    const power =
      Math.pow(Math.abs(dir.x), 10) + Math.pow(Math.abs(dir.y), 10) + Math.pow(Math.abs(dir.z), 10);
    const cubeRadius = 0.8 / Math.pow(power, 0.1);
    return 1 + (cubeRadius - 1) * morphUniform.value;
  }

  private random(): number {
    this.seed ^= this.seed << 13;
    this.seed ^= this.seed >>> 17;
    this.seed ^= this.seed << 5;
    return (this.seed >>> 0) / 0x100000000;
  }
}
