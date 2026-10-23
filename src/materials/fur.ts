import {
  Color,
  Euler,
  InstancedBufferGeometry,
  Matrix3,
  Matrix4,
  Mesh,
  ShaderMaterial,
  SphereGeometry,
  Vector3,
  Vector4,
} from 'three';
import { FUR, MAX_BRUSHES } from '../config';
import { approach, Spring, stepSprings } from '../core/spring';
import { morphUniform } from '../scene/shape';
import furFrag from '../shaders/fur.frag.glsl?raw';
import furVert from '../shaders/fur.vert.glsl?raw';
import shapeGlsl from '../shaders/shape.glsl?raw';
import type { FrameContext, Readout, Substance } from './types';

interface Brush {
  dir: Vector3;
  stroke: Vector3;
  strength: Spring;
  live: boolean;
}

/** Rough count of visible strands, for the readout. */
export function strandEstimate(density: number): number {
  return Math.round((4 * Math.PI * density * density * 0.7) / 1000);
}

/**
 * Shell-textured fur. Strokes leave springy "brushes" that bend nearby hair along the stroke;
 * pressing parts the fur; spinning makes it trail behind.
 */
export class Fur implements Substance {
  readonly id = 'fur';
  readonly accent = '#f2b77a';
  readonly poolStrength = 0.07;
  readonly hitPadding = FUR.length * 0.6;
  readonly objects;

  private readonly shells: number;
  private readonly uniforms;
  private readonly brushes: Brush[] = Array.from({ length: MAX_BRUSHES }, () => ({
    dir: new Vector3(0, 1, 0),
    stroke: new Vector3(),
    strength: new Spring(13, 4.2),
    live: false,
  }));
  private ring = 0;
  private last: Vector3 | null = null;
  private hoverLast: Vector3 | null = null;
  private press = 0;
  private pressTarget = 0;
  private readonly pressDir = new Vector3(0, 0, 1);
  private readonly delta = new Vector3();

  constructor(coarse: boolean) {
    this.shells = FUR.shells(coarse);
    const base = new SphereGeometry(1, coarse ? 96 : 128, coarse ? 64 : 96);
    const geometry = new InstancedBufferGeometry();
    geometry.setIndex(base.index);
    geometry.setAttribute('position', base.getAttribute('position'));
    geometry.instanceCount = this.shells + 1; // instance 0 is the skin under the hair

    this.uniforms = {
      uShells: { value: this.shells },
      uLen: { value: FUR.length },
      uDensity: { value: FUR.density },
      uTime: { value: 0 },
      uGrav: { value: new Vector3(0, -1, 0) },
      uOmega: { value: new Vector3() },
      uBrush: { value: Array.from({ length: MAX_BRUSHES }, () => new Vector4(0, 1, 0, 0)) },
      uBrushT: { value: Array.from({ length: MAX_BRUSHES }, () => new Vector4()) },
      uPress: { value: new Vector4(0, 0, 1, 0) },
      uRoot: { value: new Color('#2a1107') },
      uTip: { value: new Color('#f3b97f') },
      uL1: { value: new Vector3(-0.55, 0.75, 0.5).normalize() },
      uLattice: {
        value: new Matrix3().setFromMatrix4(new Matrix4().makeRotationFromEuler(new Euler(0.61, 0.37, 0.83))),
      },
      uMorph: morphUniform,
    };
    const material = new ShaderMaterial({
      uniforms: this.uniforms,
      defines: { MAX_BRUSHES },
      vertexShader: `${shapeGlsl}\n${furVert}`,
      fragmentShader: furFrag,
      alphaToCoverage: true,
    });
    const mesh = new Mesh(geometry, material);
    mesh.frustumCulled = false;
    this.objects = [mesh] as const;
  }

  hint(coarse: boolean): string {
    return coarse
      ? 'Stroke the fur to comb it. Spin it to fluff it up.'
      : 'Brush the fur with your cursor. Spin it to fluff it up.';
  }

  readout(): Readout {
    return ['Fur', `${this.shells} shells`, `≈ ${strandEstimate(FUR.density)}k strands`];
  }

  /** Records a stroke from `last` to `dir`. Returns false when the contact barely moved. */
  private stroke(dir: Vector3, last: Vector3, dt: number, factor: number): boolean {
    const delta = this.delta.subVectors(dir, last);
    delta.addScaledVector(dir, -delta.dot(dir)); // keep it tangent to the surface
    const moved = delta.length();
    if (moved < 0.004) return false;
    const speed = moved / Math.max(dt, 0.008);
    const strength = Math.min(1, 0.35 + speed * 0.12) * factor;
    delta.normalize();

    let brush = this.brushes[(this.ring + MAX_BRUSHES - 1) % MAX_BRUSHES]!;
    if (!brush.live || brush.dir.angleTo(dir) > 0.12) {
      brush = this.brushes[this.ring++ % MAX_BRUSHES]!;
      brush.stroke.copy(delta);
      brush.strength.value = 0;
    }
    brush.dir.copy(dir);
    brush.stroke.lerp(delta, 0.5).normalize();
    brush.strength.value = Math.max(brush.strength.value, strength);
    brush.strength.velocity = 0;
    brush.live = true;
    return true;
  }

  down(dir: Vector3): void {
    this.last = dir.clone();
    this.pressTarget = 1;
    this.pressDir.copy(dir);
  }

  move(dir: Vector3, dt: number): void {
    this.pressDir.copy(dir);
    if (!this.last) {
      this.last = dir.clone();
      return;
    }
    if (this.stroke(dir, this.last, dt, 1)) {
      this.pressTarget = 0.35;
      this.last.copy(dir);
    }
  }

  up(): void {
    this.pressTarget = 0;
    this.last = null;
  }

  hover(dir: Vector3, dt: number): void {
    if (!this.hoverLast) {
      this.hoverLast = dir.clone();
      return;
    }
    if (this.stroke(dir, this.hoverLast, dt, 0.6)) this.hoverLast.copy(dir);
  }

  hoverEnd(): void {
    this.hoverLast = null;
  }

  enter(): void {
    this.press = 0;
  }

  update(ctx: FrameContext): void {
    const live = this.brushes.filter((b) => b.live);
    stepSprings(
      live.map((b) => b.strength),
      ctx.dt,
      0.008,
    );
    this.brushes.forEach((b, i) => {
      if (b.live && b.strength.isSettled(1e-3, 1e-2)) {
        b.live = false;
        b.strength.value = 0;
      }
      this.uniforms.uBrush.value[i]!.set(b.dir.x, b.dir.y, b.dir.z, b.live ? b.strength.value : 0);
      this.uniforms.uBrushT.value[i]!.set(b.stroke.x, b.stroke.y, b.stroke.z, 0);
    });
    this.press += (this.pressTarget - this.press) * approach(9, ctx.dt);
    const p = this.pressDir;
    this.uniforms.uPress.value.set(p.x, p.y, p.z, this.press);
    this.uniforms.uTime.value = ctx.time;
    this.uniforms.uOmega.value.copy(ctx.omegaLocal);
    this.uniforms.uGrav.value.copy(ctx.gravityLocal);
  }
}
