import type { Vector3 } from 'three';

/** What the pointer layer needs from the app: hit-testing plus touch and spin callbacks. */
export interface PointerTarget {
  /** Unit direction (object space) under the screen point, or null if it misses the object. */
  hitTest(clientX: number, clientY: number, clampToEdge: boolean): Vector3 | null;
  touchDown(dir: Vector3): void;
  touchMove(dir: Vector3, dt: number): void;
  touchUp(): void;
  hover(dir: Vector3 | null, dt: number): void;
  spinStart(): void;
  spinDrag(dx: number, dy: number, dt: number): void;
  spinEnd(msSinceLastMove: number): void;
}

interface ActivePointer {
  id: number;
  mode: 'touch' | 'spin';
  x: number;
  y: number;
  t: number;
}

/**
 * One finger (or mouse button) at a time. Pressing on the object touches the material;
 * pressing anywhere else spins it.
 */
export class PointerInput {
  private active: ActivePointer | null = null;
  private readonly canvas: HTMLCanvasElement;
  private readonly target: PointerTarget;

  constructor(canvas: HTMLCanvasElement, target: PointerTarget) {
    this.canvas = canvas;
    this.target = target;
    canvas.addEventListener('pointerdown', this.onDown);
    canvas.addEventListener('pointermove', this.onMove);
    canvas.addEventListener('pointerup', this.onEnd);
    canvas.addEventListener('pointercancel', this.onEnd);
    canvas.addEventListener('pointerleave', () => this.target.hover(null, 0));
  }

  get spinning(): boolean {
    return this.active?.mode === 'spin';
  }

  get msSinceLastMove(): number {
    return this.active ? performance.now() - this.active.t : Infinity;
  }

  private readonly onDown = (e: PointerEvent): void => {
    if (this.active) return;
    this.canvas.setPointerCapture(e.pointerId);
    const dir = this.target.hitTest(e.clientX, e.clientY, false);
    this.active = {
      id: e.pointerId,
      mode: dir ? 'touch' : 'spin',
      x: e.clientX,
      y: e.clientY,
      t: performance.now(),
    };
    if (dir) {
      this.target.hover(null, 0);
      this.target.touchDown(dir);
    } else {
      this.canvas.classList.add('grabbing');
      this.target.spinStart();
    }
  };

  private readonly onMove = (e: PointerEvent): void => {
    const now = performance.now();
    const active = this.active;
    if (!active) {
      if (e.pointerType === 'mouse') {
        const dir = this.target.hitTest(e.clientX, e.clientY, false);
        this.canvas.classList.toggle('on-orb', dir !== null);
        this.target.hover(dir, 1 / 60);
      }
      return;
    }
    if (e.pointerId !== active.id) return;
    const dt = Math.max(1, now - active.t) / 1000;
    if (active.mode === 'spin') {
      this.target.spinDrag(e.clientX - active.x, e.clientY - active.y, dt);
    } else {
      const dir = this.target.hitTest(e.clientX, e.clientY, true);
      if (dir) this.target.touchMove(dir, dt);
    }
    active.x = e.clientX;
    active.y = e.clientY;
    active.t = now;
  };

  private readonly onEnd = (e: PointerEvent): void => {
    const active = this.active;
    if (!active || e.pointerId !== active.id) return;
    if (active.mode === 'touch') this.target.touchUp();
    else this.target.spinEnd(performance.now() - active.t);
    this.canvas.classList.remove('grabbing');
    this.active = null;
  };
}
