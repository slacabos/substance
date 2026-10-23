import { Color, Mesh, Vector3, type BufferGeometry } from 'three';
import { MAX_IMPULSES } from '../config';
import { Spring, stepSprings } from '../core/spring';
import { TrappedBubbles } from './bubbles';
import { createDeformMaterial, DeformMode, type DeformUniforms } from './deform';
import type { FrameContext, Readout, Substance } from './types';

interface Poke {
  dir: Vector3;
  depth: Spring;
  radius: number;
  live: boolean;
}

/**
 * Viscoelastic goo. Each press is a spring-driven dent; letting go springs it back and feeds
 * a global squash-and-stretch mode, which is what makes it jiggle.
 */
export class Slime implements Substance {
  readonly id = 'slime';
  readonly accent = '#9cf55a';
  readonly poolStrength = 0.16;
  readonly hitPadding = 0;
  readonly objects;

  private readonly uniforms: DeformUniforms;
  private readonly bubbles = new TrappedBubbles(11);
  private readonly pokes: Poke[] = Array.from({ length: MAX_IMPULSES }, () => ({
    dir: new Vector3(0, 1, 0),
    depth: new Spring(110, 7),
    radius: 0.35,
    live: false,
  }));
  private readonly wobble = new Spring(55, 3.2);
  private readonly wobbleAxis = new Vector3(0, 1, 0);
  private ring = 0;
  private current: Poke | null = null;
  private holdTime = 0;

  constructor(geometry: BufferGeometry) {
    const { material, uniforms } = createDeformMaterial(
      {
        color: new Color(0.55, 1.0, 0.2),
        metalness: 0,
        roughness: 0.15,
        transmission: 0.88,
        thickness: 1.6,
        ior: 1.46,
        attenuationColor: new Color(0.35, 0.95, 0.12),
        attenuationDistance: 1.1,
        emissive: new Color(0.06, 0.2, 0.02),
        clearcoat: 1,
        clearcoatRoughness: 0.05,
        sheen: 0.35,
        sheenColor: new Color(0.75, 1.0, 0.55),
        sheenRoughness: 0.45,
        envMapIntensity: 1.1,
      },
      DeformMode.Slime,
    );
    const mesh = new Mesh(geometry, material);
    mesh.frustumCulled = false;
    this.uniforms = uniforms;
    this.objects = [mesh, this.bubbles.mesh] as const;
  }

  hint(): string {
    return 'Press and hold to squish. Let go and it jiggles.';
  }

  readout(): Readout {
    return ['Slime', 'n ≈ 1.46', 'viscoelastic'];
  }

  private newPoke(dir: Vector3): Poke {
    const p = this.pokes[this.ring++ % MAX_IMPULSES]!;
    p.dir.copy(dir);
    p.depth.value = 0;
    p.depth.velocity = -0.6;
    p.depth.target = -0.13;
    p.radius = 0.3 + Math.random() * 0.07;
    p.live = true;
    return p;
  }

  private release(p: Poke, strength: number): void {
    p.depth.target = 0;
    this.wobbleAxis.copy(p.dir);
    this.wobble.velocity += p.depth.value * 5 * strength;
  }

  down(dir: Vector3): void {
    this.current = this.newPoke(dir);
    this.holdTime = 0;
  }

  move(dir: Vector3): void {
    if (!this.current) return;
    if (dir.angleTo(this.current.dir) > 0.24) {
      // dragged far: let the old dent go and start a new one, leaving a springy trail
      const prev = this.current;
      this.release(prev, 0.5);
      this.current = this.newPoke(dir);
      this.current.depth.value = prev.depth.value * 0.6;
    } else {
      this.current.dir.lerp(dir, 0.35).normalize();
    }
  }

  up(): void {
    if (this.current) this.release(this.current, 1);
    this.current = null;
  }

  enter(): void {
    this.kick(new Vector3(0, 1, 0), 1.4);
  }

  onShapeChange(): void {
    this.kick(new Vector3(0, 1, 0), 1.2);
  }

  onSpinJerk(axisLocal: Vector3, magnitude: number): void {
    this.kick(axisLocal, Math.min(magnitude, 6) * 0.06);
  }

  private kick(axis: Vector3, amount: number): void {
    this.wobbleAxis.copy(axis);
    this.wobble.velocity -= amount;
  }

  update(ctx: FrameContext): void {
    if (this.current) {
      this.holdTime += ctx.dt;
      this.current.depth.target = -0.13 - 0.14 * Math.min(1, this.holdTime / 1.2);
    }
    stepSprings([this.wobble, ...this.pokes.filter((p) => p.live).map((p) => p.depth)], ctx.dt);
    this.wobble.value = Math.min(0.22, Math.max(-0.22, this.wobble.value));

    this.pokes.forEach((p, i) => {
      if (p.live && p !== this.current && p.depth.target === 0 && p.depth.isSettled()) {
        p.live = false;
        p.depth.value = 0;
      }
      this.uniforms.imp[i]!.set(p.dir.x, p.dir.y, p.dir.z, p.live ? p.depth.value : 0);
      this.uniforms.impB[i]!.x = p.radius;
    });
    const a = this.wobbleAxis;
    this.uniforms.uWob.value.set(a.x, a.y, a.z, this.wobble.value);
    this.uniforms.uTime.value = ctx.time;
    this.bubbles.update(ctx.time, 1 + this.wobble.value * 0.8);
  }
}
