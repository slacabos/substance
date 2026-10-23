import { Quaternion, Vector3 } from 'three';
import { approach } from '../core/spring';

const DRAG_RADIANS_PER_PX = 0.0065;
const MAX_FLICK = 14;

/**
 * Spins a quaternion: drag to rotate, flick for inertia, then it eases back to a slow idle spin.
 * Also keeps a lagged copy of the angular velocity for things that trail behind (fur).
 */
export class SpinController {
  readonly angularVelocity = new Vector3();
  readonly laggedVelocity = new Vector3();
  private readonly idle: Vector3;
  private readonly lagVelocity = new Vector3();
  private readonly previous = new Vector3();
  private readonly dq = new Quaternion();
  private readonly axis = new Vector3();
  private readonly tmp = new Vector3();
  private readonly quaternion: Quaternion;

  constructor(quaternion: Quaternion, idleSpeed: number) {
    this.quaternion = quaternion;
    this.idle = new Vector3(0, idleSpeed, 0);
    this.angularVelocity.copy(this.idle);
  }

  grab(): void {
    this.angularVelocity.set(0, 0, 0);
  }

  /** Rotates by a screen-space drag of (dx, dy) pixels over `dt` seconds. */
  drag(dx: number, dy: number, dt: number): void {
    // dragging right turns about +y, dragging down turns the front face downward (about +x)
    this.axis.set(dy, dx, 0);
    const angle = this.axis.length() * DRAG_RADIANS_PER_PX;
    if (angle <= 0) return;
    this.axis.normalize();
    this.dq.setFromAxisAngle(this.axis, angle);
    this.quaternion.premultiply(this.dq);
    this.angularVelocity.lerp(this.axis.multiplyScalar(Math.min(angle / dt, MAX_FLICK)), 0.6);
  }

  /** Call on release. A finger that stopped before lifting shouldn't fling the object. */
  release(msSinceLastMove: number): void {
    if (msSinceLastMove > 90) this.angularVelocity.set(0, 0, 0);
  }

  /**
   * Advances one frame. While held, the velocity bleeds off if the finger stays still.
   * Returns the change in angular velocity this frame (world space), which jiggly materials react to.
   */
  update(dt: number, held: boolean, msSinceLastMove: number): Vector3 {
    if (!held) {
      this.angularVelocity.lerp(this.idle, approach(1.1, dt));
      const angle = this.angularVelocity.length() * dt;
      if (angle > 1e-7) {
        this.dq.setFromAxisAngle(this.tmp.copy(this.angularVelocity).normalize(), angle);
        this.quaternion.premultiply(this.dq);
      }
    } else if (msSinceLastMove > 60) {
      this.angularVelocity.multiplyScalar(Math.exp(-dt * 12));
    }

    // springy lag: overshoots a little when the spin stops, so hair swings past and settles
    this.lagVelocity
      .addScaledVector(this.tmp.subVectors(this.angularVelocity, this.laggedVelocity), 30 * dt)
      .multiplyScalar(Math.exp(-dt * 5));
    this.laggedVelocity.addScaledVector(this.lagVelocity, dt);
    if (this.laggedVelocity.length() > 5) this.laggedVelocity.setLength(5);

    const jerk = this.tmp.subVectors(this.angularVelocity, this.previous);
    this.previous.copy(this.angularVelocity);
    return jerk;
  }
}
