import { Quaternion, Vector3 } from 'three';
import { intersectShape } from './core/hit';
import { Spring, stepSprings } from './core/spring';
import type { DeviceProfile } from './device';
import { createDeformGeometry } from './materials/deform';
import { Fur } from './materials/fur';
import { Ocean } from './materials/ocean';
import { Slime } from './materials/slime';
import type { FrameContext, MaterialId, ShapeId, Substance } from './materials/types';
import { Water } from './materials/water';
import { PointerInput, type PointerTarget } from './interaction/pointer';
import { SpinController } from './interaction/spin';
import { ShapeMorph } from './scene/shape';
import { Stage } from './scene/stage';
import { Controls } from './ui/controls';

export class App implements PointerTarget {
  private readonly stage: Stage;
  private readonly device: DeviceProfile;
  private readonly materials: Record<MaterialId, Substance>;
  private readonly controls: Controls;
  private readonly pointer: PointerInput;
  private readonly spin: SpinController;
  private readonly morph = new ShapeMorph();
  /** Scale "pop" when switching material. */
  private readonly pop = new Spring(240, 13, 1, 1);
  private current: Substance;
  private time = 0;
  private last = performance.now();

  private readonly ctx: FrameContext;
  private readonly invQ = new Quaternion();
  private readonly tmp = new Vector3();
  private readonly rayOrigin = new Vector3();
  private readonly rayDir = new Vector3();

  constructor(canvas: HTMLCanvasElement, device: DeviceProfile) {
    this.device = device;
    this.stage = new Stage(canvas);
    const geometry = createDeformGeometry(device.coarse);
    this.materials = {
      water: new Water(geometry, device.coarse),
      ocean: new Ocean(geometry),
      slime: new Slime(geometry),
      fur: new Fur(device.coarse),
    };
    for (const m of Object.values(this.materials)) {
      this.stage.anchor.add(...m.objects);
      if (m.worldObjects) this.stage.scene.add(...m.worldObjects);
    }
    this.current = this.materials.water;

    this.spin = new SpinController(this.stage.anchor.quaternion, device.reduceMotion ? 0.06 : 0.2);
    this.ctx = {
      time: 0,
      dt: 0,
      anchor: this.stage.anchor,
      omegaLocal: new Vector3(),
      gravityLocal: new Vector3(0, -1, 0),
    };

    this.controls = new Controls(
      { onMaterial: (id) => this.setMaterial(id), onShape: (id) => this.setShape(id) },
      device.coarse,
    );
    this.pointer = new PointerInput(canvas, this);

    const resize = () => {
      const topInset = this.controls.top.getBoundingClientRect().bottom + 8;
      const bottomInset = window.innerHeight - this.controls.dock.getBoundingClientRect().top + 8;
      this.stage.fit(window.innerWidth, window.innerHeight, topInset, bottomInset);
    };
    window.addEventListener('resize', resize);
    new ResizeObserver(resize).observe(this.controls.dock);
    resize();

    this.controls.showShape('orb');
    this.setMaterial('water', false);
    this.materials.water.enter(new Vector3(0, 0, 1));
  }

  start(): void {
    const frame = (now: number) => {
      this.tick(now);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  setMaterial(id: MaterialId, animate = true): void {
    this.current.hoverEnd?.();
    this.current = this.materials[id];
    for (const m of Object.values(this.materials)) {
      const visible = m === this.current;
      for (const o of m.objects) o.visible = visible;
      for (const o of m.worldObjects ?? []) o.visible = visible;
    }
    this.stage.pool.set(this.current.accent, this.current.poolStrength);
    this.controls.showMaterial(id, this.current.hint(this.device.coarse));
    this.controls.showReadout(this.current.readout());
    if (animate) {
      this.pop.value = this.device.reduceMotion ? 0.97 : 0.84;
      this.pop.velocity = 0;
      this.current.enter(this.frontDirection());
    }
  }

  setShape(id: ShapeId): void {
    if (id === this.morph.shape) return;
    this.morph.set(id, this.device.reduceMotion);
    this.controls.showShape(id);
    this.current.onShapeChange?.(this.frontDirection());
  }

  // ---- PointerTarget ----

  hitTest(clientX: number, clientY: number, clampToEdge: boolean): Vector3 | null {
    const { camera, anchor, renderer } = this.stage;
    const rect = renderer.domElement.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((clientY - rect.top) / rect.height) * 2 + 1;
    this.rayOrigin.setFromMatrixPosition(camera.matrixWorld);
    this.rayDir.set(x, y, 0.5).unproject(camera).sub(this.rayOrigin).normalize();
    // into object space: a unit-sized shape at the origin
    this.invQ.copy(anchor.quaternion).invert();
    this.rayOrigin.sub(anchor.position).applyQuaternion(this.invQ).divideScalar(anchor.scale.x);
    this.rayDir.applyQuaternion(this.invQ);
    return intersectShape(
      this.rayOrigin,
      this.rayDir,
      this.morph.value,
      this.current.hitPadding,
      clampToEdge,
    );
  }

  touchDown(dir: Vector3): void {
    this.current.down(dir);
  }
  touchMove(dir: Vector3, dt: number): void {
    this.current.move(dir, dt);
  }
  touchUp(): void {
    this.current.up();
  }
  hover(dir: Vector3 | null, dt: number): void {
    if (dir) this.current.hover?.(dir, dt);
    else this.current.hoverEnd?.();
  }
  spinStart(): void {
    this.spin.grab();
  }
  spinDrag(dx: number, dy: number, dt: number): void {
    this.spin.drag(dx, dy, dt);
  }
  spinEnd(msSinceLastMove: number): void {
    this.spin.release(msSinceLastMove);
  }

  // ---- frame loop ----

  /** The point on the object facing the camera, in object space. */
  private frontDirection(): Vector3 {
    const { camera, anchor } = this.stage;
    this.invQ.copy(anchor.quaternion).invert();
    return this.tmp.copy(camera.position).sub(anchor.position).normalize().applyQuaternion(this.invQ).clone();
  }

  private tick(now: number): void {
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    this.time += dt;
    const { anchor } = this.stage;

    const jerk = this.spin.update(dt, this.pointer.spinning, this.pointer.msSinceLastMove);
    this.invQ.copy(anchor.quaternion).invert();
    const jerkSize = jerk.length();
    if (jerkSize > 0.4) this.current.onSpinJerk?.(jerk.normalize().applyQuaternion(this.invQ), jerkSize);

    this.morph.update(dt);
    anchor.position.y = Math.sin(this.time * 0.8) * (this.device.reduceMotion ? 0.01 : 0.04);
    stepSprings([this.pop], dt, 0.008);
    anchor.scale.setScalar(this.pop.value);

    this.ctx.time = this.time;
    this.ctx.dt = dt;
    this.ctx.omegaLocal.copy(this.spin.laggedVelocity).applyQuaternion(this.invQ);
    this.ctx.gravityLocal.set(0, -1, 0).applyQuaternion(this.invQ);
    // keep every material's simulation running so switching back mid-ripple looks continuous
    for (const m of Object.values(this.materials)) m.update(this.ctx);

    this.controls.showReadout(this.current.readout());
    this.stage.render();
  }
}
