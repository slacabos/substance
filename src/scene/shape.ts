import type { IUniform } from 'three';
import { Spring, stepSprings } from '../core/spring';
import type { ShapeId } from '../materials/types';

/** Shared by every material: 0 = sphere, 1 = rounded cube. A little overshoot is allowed for bounce. */
export const morphUniform: IUniform<number> = { value: 0 };

/** Springs the morph uniform toward the chosen shape. */
export class ShapeMorph {
  private readonly spring = new Spring(70, 9);
  shape: ShapeId = 'orb';

  set(shape: ShapeId, instant: boolean): void {
    this.shape = shape;
    this.spring.target = shape === 'cube' ? 1 : 0;
    if (instant) {
      this.spring.value = this.spring.target;
      this.spring.velocity = 0;
    }
  }

  get value(): number {
    return this.spring.value;
  }

  update(dt: number): void {
    stepSprings([this.spring], dt, 0.01);
    this.spring.value = Math.min(1.15, Math.max(-0.15, this.spring.value));
    morphUniform.value = this.spring.value;
  }
}
