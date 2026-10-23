import { Vector3 } from 'three';

export const OCEAN_WEATHER_PATCHES = 4;

export interface WeatherPatch {
  readonly center: Vector3;
  strength: number;
  age: number;
}

export function weatherRadius(age: number): number {
  return 0.35 + 0.3 * Math.min(1, Math.max(0, age) / 5);
}

/** A few moving, fading regions of rough water, stored in the orb's local coordinates. */
export class OceanWeather {
  readonly patches: WeatherPatch[] = Array.from({ length: OCEAN_WEATHER_PATCHES }, () => ({
    center: new Vector3(0, 1, 0),
    strength: 0,
    age: 0,
  }));

  private readonly lastDir = new Vector3(0, 0, 1);
  private active = -1;
  private travel = 0;
  private churn = 0;

  down(dir: Vector3): void {
    this.lastDir.copy(dir);
    this.active = this.nextSlot();
    const patch = this.patches[this.active]!;
    patch.center.copy(dir);
    patch.strength = 0;
    patch.age = 0;
    this.travel = 0;
    this.churn = 0;
  }

  move(dir: Vector3, dt: number): void {
    const distance = this.lastDir.angleTo(dir);
    this.lastDir.copy(dir);
    if (distance < 1e-4) return;

    const speed = distance / Math.max(dt, 1 / 120);
    if (speed < 0.35) return;
    const energy = Math.min(1, speed / 2.5);
    this.churn = Math.min(1, this.churn + energy * dt * 0.8);
    this.travel += distance;

    if (this.active < 0 || this.travel >= 0.22 || this.patches[this.active]!.age >= 0.35) {
      this.active = this.nextSlot();
      const patch = this.patches[this.active]!;
      patch.center.copy(dir);
      patch.strength = Math.max(0.2, this.churn);
      patch.age = 0;
      this.travel = 0;
    } else {
      const patch = this.patches[this.active]!;
      patch.center.lerp(dir, Math.min(1, distance * 3)).normalize();
      patch.strength = Math.min(1, Math.max(patch.strength, this.churn));
    }
  }

  up(): void {
    this.active = -1;
  }

  clear(): void {
    for (const patch of this.patches) {
      patch.strength = 0;
      patch.age = 0;
    }
    this.active = -1;
    this.travel = 0;
    this.churn = 0;
  }

  update(dt: number): void {
    this.churn *= Math.exp(-dt / 5);
    for (const patch of this.patches) {
      if (patch.strength <= 0) continue;
      patch.age += dt;
      patch.strength *= Math.exp(-dt / 5);
      if (patch.strength < 0.01) patch.strength = 0;
    }
  }

  get peak(): number {
    return Math.max(...this.patches.map((patch) => patch.strength));
  }

  strengthAt(dir: Vector3): number {
    let strength = 0;
    for (const patch of this.patches) {
      if (patch.strength <= 0) continue;
      const radius = weatherRadius(patch.age);
      const dot = patch.center.dot(dir);
      const t = Math.min(
        1,
        Math.max(0, (dot - Math.cos(radius)) / (Math.cos(radius * 0.4) - Math.cos(radius))),
      );
      strength = Math.max(strength, patch.strength * t * t * (3 - 2 * t));
    }
    return strength;
  }

  private nextSlot(): number {
    const empty = this.patches.findIndex((patch) => patch.strength === 0);
    if (empty >= 0) return empty;
    let weakest = 0;
    for (let i = 1; i < this.patches.length; i++) {
      if (this.patches[i]!.strength < this.patches[weakest]!.strength) weakest = i;
    }
    return weakest;
  }
}
