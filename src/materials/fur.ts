import {
  BufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  SphereGeometry,
  Vector3,
  type Object3D,
} from 'three';
import { MeshBasicNodeMaterial, type WebGPURenderer } from 'three/webgpu';
import { FUR } from '../config';
import { approach } from '../core/spring';
import { createFurUniforms, createSkinNodes, createStrandNodes } from '../nodes/fur';
import { FurSim } from './furSim';
import type { FrameContext, Readout, Substance } from './types';

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
 * Fur made of individual strands over a dark skin, simulated on the GPU (see FurSim): hair has weight and
 * momentum, trails behind a spin, stays combed where it's stroked and parts where it's pressed.
 */
export class Fur implements Substance {
  readonly id = 'fur';
  readonly accent = '#f2b77a';
  readonly poolStrength = 0.07;
  readonly hitPadding = FUR.length * 0.6;
  readonly objects;
  readonly worldObjects;

  private readonly sim: FurSim;
  private readonly strands: Mesh;
  private readonly anchor: Object3D;
  private last: Vector3 | null = null;
  private hoverLast: Vector3 | null = null;
  private press = 0;
  private pressTarget = 0;
  private readonly delta = new Vector3();

  constructor(coarse: boolean, renderer: WebGPURenderer, anchor: Object3D) {
    this.anchor = anchor;
    const { roots, rand } = createRoots(strandCount(FUR.density));
    this.sim = new FurSim(renderer, roots, rand, FUR.points(coarse));
    const uniforms = createFurUniforms();

    const geometry = createStrandTemplate(FUR.segments(coarse));
    geometry.instanceCount = this.sim.count;
    const strandNodes = createStrandNodes(uniforms, this.sim);
    // with MSAA, faint coverage becomes a share of each pixel's samples, so strands need no sorting
    const strandMaterial = new MeshBasicNodeMaterial({ alphaToCoverage: true });
    strandMaterial.positionNode = strandNodes.position;
    strandMaterial.colorNode = strandNodes.color;
    strandMaterial.opacityNode = strandNodes.opacity;
    this.strands = new Mesh(geometry, strandMaterial);
    this.strands.frustumCulled = false;
    this.strands.renderOrder = 1;

    const skinNodes = createSkinNodes(uniforms);
    const skinMaterial = new MeshBasicNodeMaterial();
    skinMaterial.positionNode = skinNodes.position;
    skinMaterial.colorNode = skinNodes.color;
    const skin = new Mesh(new SphereGeometry(1, coarse ? 96 : 128, coarse ? 64 : 96), skinMaterial);
    skin.frustumCulled = false;

    this.objects = [skin] as const;
    // the simulated strands are in world space, so they hang off the scene rather than the spinning anchor
    this.worldObjects = [this.strands] as const;
    this.sim.reset(anchor);
  }

  hint(coarse: boolean): string {
    return coarse
      ? 'Stroke the fur to comb it. Spin it to fluff it up.'
      : 'Brush the fur with your cursor. Spin it to fluff it up.';
  }

  readout(): Readout {
    return ['Fur', `≈ ${strandEstimate(FUR.density)}k strands`, 'keratin n ≈ 1.55'];
  }

  /** Combs along a stroke from `last` to `dir`. Returns false when the contact barely moved. */
  private stroke(dir: Vector3, last: Vector3, dt: number, factor: number): boolean {
    const delta = this.delta.subVectors(dir, last);
    delta.addScaledVector(dir, -delta.dot(dir)); // keep it tangent to the surface
    const moved = delta.length();
    if (moved < 0.004) return false;
    const speed = moved / Math.max(dt, 0.008);
    const strength = Math.min(1, 0.35 + speed * 0.12) * factor;
    const u = this.sim.uniforms;
    delta.normalize();
    u.brushDir.value.copy(dir);
    u.brushStroke.value.copy(delta).multiplyScalar(strength);
    u.brushVel.value.copy(delta).multiplyScalar(Math.min(speed, 6) * factor);
    u.brushAmount.value = 1;
    return true;
  }

  down(dir: Vector3): void {
    this.last = dir.clone();
    this.pressTarget = 1;
    this.sim.uniforms.pressDir.value.copy(dir);
  }

  move(dir: Vector3, dt: number): void {
    this.sim.uniforms.pressDir.value.copy(dir);
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
    // the simulation pauses while the fur is hidden, so start it fresh rather than from stale positions
    this.sim.reset(this.anchor);
  }

  update(ctx: FrameContext): void {
    const u = this.sim.uniforms;
    u.brushAmount.value *= 1 - approach(12, ctx.dt); // a stroke only combs while it keeps moving
    this.press += (this.pressTarget - this.press) * approach(9, ctx.dt);
    u.press.value = this.press;
    if (this.strands.visible) this.sim.update(this.anchor, ctx.time, ctx.dt);
  }
}
