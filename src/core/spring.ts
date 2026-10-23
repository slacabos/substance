/** A damped spring that pulls `value` toward `target`. Integrate with small fixed steps via {@link stepSprings}. */
export class Spring {
  value: number;
  velocity = 0;
  target: number;
  stiffness: number;
  damping: number;

  constructor(stiffness: number, damping: number, value = 0, target = value) {
    this.stiffness = stiffness;
    this.damping = damping;
    this.value = value;
    this.target = target;
  }

  step(h: number): void {
    this.velocity += (this.stiffness * (this.target - this.value) - this.damping * this.velocity) * h;
    this.value += this.velocity * h;
  }

  isSettled(valueEps = 2e-4, velocityEps = 2e-3): boolean {
    return Math.abs(this.value - this.target) < valueEps && Math.abs(this.velocity) < velocityEps;
  }
}

/** Splits a frame into equal substeps no longer than `maxStep`, so stiff springs stay stable at low frame rates. */
export function substeps(dt: number, maxStep: number): { count: number; h: number } {
  const count = Math.max(1, Math.ceil(dt / maxStep));
  return { count, h: dt / count };
}

export function stepSprings(springs: Iterable<Spring>, dt: number, maxStep = 0.006): void {
  const list = [...springs];
  const { count, h } = substeps(dt, maxStep);
  for (let i = 0; i < count; i++) for (const s of list) s.step(h);
}

/** Frame-rate independent exponential approach: the fraction to move toward a target this frame. */
export function approach(rate: number, dt: number): number {
  return 1 - Math.exp(-rate * dt);
}
