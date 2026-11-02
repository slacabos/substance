import { uniform } from 'three/tsl';
import { unfurlSolidWeight } from '../core/transition';
import { Spring, stepSprings } from '../core/spring';
import type { ShapeId } from '../materials/types';

/** Shared by every material: 0 = sphere, 1 = rounded cube. A little overshoot is allowed for bounce. */
export const morphUniform = uniform(0);
/**
 * 0 = the orb or cube, 1 = laid out flat as endless ground (core/chart.ts). Never overshoots: past 1 the far ground
 * would run off past the horizon.
 */
export const planeUniform = uniform(0);

/** Springs the morph uniforms toward the chosen shape. */
export class ShapeMorph {
  private readonly spring = new Spring(70, 9);
  // A critically damped opening: enough time to read the unfurl, with no ground overshoot.
  private readonly flat = new Spring(9, 6);
  shape: ShapeId = 'orb';

  set(shape: ShapeId, instant: boolean): void {
    this.shape = shape;
    // Remember the source solid while the plane is open; rounding follows the unfolding timeline.
    if (shape !== 'plane') this.spring.target = shape === 'cube' ? 1 : 0;
    this.flat.target = shape === 'plane' ? 1 : 0;
    if (instant) {
      for (const s of [this.spring, this.flat]) {
        s.value = s.target;
        s.velocity = 0;
      }
    }
  }

  get value(): number {
    return unfurlSolidWeight(this.spring.value, this.flat.value);
  }

  /** How far toward the plane, 0 to 1. */
  get plane(): number {
    return this.flat.value;
  }

  update(dt: number): void {
    stepSprings([this.spring, this.flat], dt, 0.01);
    this.spring.value = Math.min(1.15, Math.max(-0.15, this.spring.value));
    this.flat.value = Math.min(1, Math.max(0, this.flat.value));
    // settled, it lands exactly: the orb and the cube test the plane's weight against 0 (drawing hair's copies,
    // the water's bed), and a spring only ever creeps toward it
    if (this.flat.isSettled()) {
      this.flat.value = this.flat.target;
      this.flat.velocity = 0;
    }
    morphUniform.value = this.value;
    planeUniform.value = this.flat.value;
  }
}
