import {
  BufferAttribute,
  Color,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  MathUtils,
  Mesh,
  ShaderMaterial,
  SphereGeometry,
  Vector2,
  Vector3,
  Vector4,
  type PerspectiveCamera,
} from 'three';
import { FUR, MAX_BRUSHES } from '../config';
import { approach, Spring, stepSprings } from '../core/spring';
import { morphUniform } from '../scene/shape';
import skinFrag from '../shaders/fur-skin.frag.glsl?raw';
import skinVert from '../shaders/fur-skin.vert.glsl?raw';
import strandFrag from '../shaders/fur-strand.frag.glsl?raw';
import strandVert from '../shaders/fur-strand.vert.glsl?raw';
import shapeGlsl from '../shaders/shape.glsl?raw';
import type { FrameContext, Readout, Substance } from './types';

interface Brush {
  dir: Vector3;
  stroke: Vector3;
  strength: Spring;
  live: boolean;
}

/** How many strands cover the skin at a given density (strands per unit length). */
export function strandCount(density: number): number {
  return Math.round(4 * Math.PI * density * density * 0.7);
}

/** Strand count in thousands, for the readout. */
export function strandEstimate(density: number): number {
  return Math.round(strandCount(density) / 1000);
}

/** A ribbon from root (y = 0) to tip (y = 1), with x = -1 / 1 on its two sides. */
function createStrandTemplate(segments: number): InstancedBufferGeometry {
  const geometry = new InstancedBufferGeometry();
  const position = new Float32Array((segments + 1) * 2 * 3);
  const index: number[] = [];
  for (let i = 0; i <= segments; i++) {
    position.set([-1, i / segments, 0, 1, i / segments, 0], i * 6);
    if (i < segments) index.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
  }
  geometry.setAttribute('position', new BufferAttribute(position, 3));
  geometry.setIndex(index);
  return geometry;
}

/** Evenly spread, slightly jittered root directions on the unit sphere (a jittered Fibonacci spiral). */
function createRoots(count: number): { roots: Float32Array; rand: Float32Array } {
  const roots = new Float32Array(count * 3);
  const rand = new Float32Array(count * 3);
  const golden = Math.PI * (3 - Math.sqrt(5));
  const jitter = 1.2 / Math.sqrt(count);
  const v = new Vector3();
  for (let i = 0; i < count; i++) {
    const y = 1 - (2 * (i + 0.5)) / count;
    const r = Math.sqrt(1 - y * y);
    const phi = i * golden;
    v.set(Math.cos(phi) * r, y, Math.sin(phi) * r);
    v.x += (Math.random() - 0.5) * jitter;
    v.y += (Math.random() - 0.5) * jitter;
    v.z += (Math.random() - 0.5) * jitter;
    v.normalize().toArray(roots, i * 3);
    rand.set([Math.random(), Math.random(), Math.random()], i * 3);
  }
  return { roots, rand };
}

/**
 * Fur made of individual strands over a dark skin. Strokes leave springy "brushes" that bend nearby hair
 * along the stroke; pressing parts the fur; spinning makes it trail behind.
 */
export class Fur implements Substance {
  readonly id = 'fur';
  readonly accent = '#f2b77a';
  readonly poolStrength = 0.07;
  readonly hitPadding = FUR.length * 0.6;
  readonly objects;

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
  private readonly bufferSize = new Vector2();

  constructor(coarse: boolean) {
    const count = strandCount(FUR.density);
    const geometry = createStrandTemplate(FUR.segments(coarse));
    const { roots, rand } = createRoots(count);
    geometry.setAttribute('aRoot', new InstancedBufferAttribute(roots, 3));
    geometry.setAttribute('aRand', new InstancedBufferAttribute(rand, 3));
    geometry.instanceCount = count;

    const light = {
      uRoot: { value: new Color('#2a1107') },
      uTip: { value: new Color('#f3b97f') },
      uL1: { value: new Vector3(-0.55, 0.75, 0.5).normalize() },
      uMorph: morphUniform,
    };
    this.uniforms = {
      ...light,
      uLen: { value: FUR.length },
      uDensity: { value: FUR.density },
      uTime: { value: 0 },
      uGrav: { value: new Vector3(0, -1, 0) },
      uOmega: { value: new Vector3() },
      uBrush: { value: Array.from({ length: MAX_BRUSHES }, () => new Vector4(0, 1, 0, 0)) },
      uBrushT: { value: Array.from({ length: MAX_BRUSHES }, () => new Vector4()) },
      uPress: { value: new Vector4(0, 0, 1, 0) },
      uCamera: { value: new Vector3() },
      uPixel: { value: 0.001 },
    };

    const strands = new Mesh(
      geometry,
      new ShaderMaterial({
        uniforms: this.uniforms,
        defines: { MAX_BRUSHES },
        vertexShader: `${shapeGlsl}\n${strandVert}`,
        fragmentShader: strandFrag,
        // with MSAA, faint coverage becomes a share of each pixel's samples, so strands need no sorting
        alphaToCoverage: true,
      }),
    );
    strands.frustumCulled = false;
    strands.renderOrder = 1;
    // Strands face the camera and never get thinner than a pixel, so they need its position and pixel size.
    strands.onBeforeRender = (renderer, _scene, camera) => {
      const { uCamera, uPixel } = this.uniforms;
      strands.worldToLocal(uCamera.value.copy(camera.position));
      const fov = MathUtils.degToRad((camera as PerspectiveCamera).fov);
      uPixel.value = (2 * Math.tan(fov / 2)) / renderer.getDrawingBufferSize(this.bufferSize).y;
    };

    const skin = new Mesh(
      new SphereGeometry(1, coarse ? 96 : 128, coarse ? 64 : 96),
      new ShaderMaterial({
        uniforms: light,
        vertexShader: `${shapeGlsl}\n${skinVert}`,
        fragmentShader: skinFrag,
      }),
    );
    skin.frustumCulled = false;

    this.objects = [skin, strands] as const;
  }

  hint(coarse: boolean): string {
    return coarse
      ? 'Stroke the fur to comb it. Spin it to fluff it up.'
      : 'Brush the fur with your cursor. Spin it to fluff it up.';
  }

  readout(): Readout {
    return ['Fur', `≈ ${strandEstimate(FUR.density)}k strands`, 'keratin n ≈ 1.55'];
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
