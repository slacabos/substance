import { Vector3 } from 'three';
import { MAX_IMPULSES } from '../config';
import { approach } from '../core/spring';
import type { DeformUniforms } from './deform';

/**
 * Ripple sources for the water-like materials. Each tap or drag step writes a source into a ring buffer
 * of shader uniforms; the shader turns every source into an expanding, decaying ring.
 */
export class RippleField {
  private ring = 0;
  private press = 0;
  private pressTarget = 0;
  private lastSpawn = -Infinity;
  private readonly lastDir = new Vector3(0, 0, 1);
  private readonly pressDir = new Vector3(0, 0, 1);
  private readonly uniforms: DeformUniforms;

  constructor(uniforms: DeformUniforms) {
    this.uniforms = uniforms;
  }

  spawn(dir: Vector3, strength: number, time: number): void {
    const i = this.ring++ % MAX_IMPULSES;
    this.uniforms.uImp.value[i]!.set(dir.x, dir.y, dir.z, time);
    this.uniforms.uImpB.value[i]!.x = strength;
  }

  down(dir: Vector3, time: number): void {
    this.spawn(dir, 1, time);
    this.pressTarget = 1;
    this.pressDir.copy(dir);
    this.lastDir.copy(dir);
    this.lastSpawn = time;
  }

  /** Returns how far (radians) the contact travelled when it spawned a ripple, 0 otherwise. */
  move(dir: Vector3, time: number): number {
    this.pressDir.copy(dir);
    const moved = dir.angleTo(this.lastDir);
    if (moved > 0.12 || time - this.lastSpawn > 0.35) {
      this.spawn(dir, Math.min(1, 0.35 + moved * 2.5), time);
      this.lastDir.copy(dir);
      this.lastSpawn = time;
      return moved;
    }
    return 0;
  }

  up(time: number): void {
    this.pressTarget = 0;
    this.spawn(this.pressDir, 0.55, time);
  }

  update(time: number, dt: number): void {
    this.press += (this.pressTarget - this.press) * approach(10, dt);
    const p = this.pressDir;
    this.uniforms.uPress.value.set(p.x, p.y, p.z, this.press);
    this.uniforms.uTime.value = time;
  }
}
