import { Mesh, type BufferGeometry, type Vector3 } from 'three';
import { approach } from '../core/spring';
import { createDeformMaterial, DeformMode, type DeformUniforms } from './deform';
import { RippleField } from './ripples';
import type { FrameContext, Readout, Substance } from './types';

const MAX_STORM = 2.4;

/** Storm level (1 = calm) to a Beaufort number for the readout. */
export function beaufortFor(storm: number): number {
  return Math.round(2 + (storm - 1) * 4);
}

/** Layered swells with foam on the crests. Stirring it builds a storm that calms down over time. */
export class Ocean implements Substance {
  readonly id = 'ocean';
  readonly accent = '#2fb8b0';
  readonly poolStrength = 0.14;
  readonly hitPadding = 0;
  readonly objects;

  private readonly ripples;
  private readonly uniforms: DeformUniforms;
  private storm = 1;
  private time = 0;

  constructor(geometry: BufferGeometry) {
    const { material, uniforms } = createDeformMaterial(
      {
        color: 0xffffff,
        metalness: 0,
        roughness: 0.09,
        specularIntensity: 1,
        ior: 1.34,
        clearcoat: 0.4,
        clearcoatRoughness: 0.12,
        envMapIntensity: 1.25,
      },
      DeformMode.Ocean,
    );
    const mesh = new Mesh(geometry, material);
    mesh.frustumCulled = false;
    this.uniforms = uniforms;
    this.ripples = new RippleField(uniforms);
    this.objects = [mesh] as const;
  }

  hint(): string {
    return 'Tap to drop a stone. Churn it to whip up a storm.';
  }

  readout(): Readout {
    return ['Ocean', `Beaufort ${beaufortFor(this.storm)}`, '35 g/kg salt'];
  }

  down(dir: Vector3): void {
    this.ripples.down(dir, this.time);
    this.storm = Math.min(MAX_STORM, this.storm + 0.12);
  }
  move(dir: Vector3): void {
    const moved = this.ripples.move(dir, this.time);
    this.storm = Math.min(MAX_STORM, this.storm + moved * 0.35);
  }
  up(): void {
    this.ripples.up(this.time);
  }
  enter(front: Vector3): void {
    this.ripples.spawn(front, 0.9, this.time);
  }
  onShapeChange(front: Vector3): void {
    this.ripples.spawn(front, 0.7, this.time);
  }

  update(ctx: FrameContext): void {
    this.time = ctx.time;
    this.ripples.update(ctx.time, ctx.dt);
    this.storm += (1 - this.storm) * approach(0.18, ctx.dt);
    this.uniforms.uStorm.value = this.storm;
  }
}
