import { Color, Group, Mesh, type BufferGeometry, type Vector3 } from 'three';
import { RisingBubbles } from './bubbles';
import { createDeformMaterial, DeformMode } from './deform';
import { RippleField } from './ripples';
import type { FrameContext, Readout, Substance } from './types';

export class Water implements Substance {
  readonly id = 'water';
  readonly accent = '#8fd3ff';
  readonly poolStrength = 0.2;
  readonly hitPadding = 0;
  readonly objects;
  readonly worldObjects;

  private readonly ripples;
  private readonly bubbles = new RisingBubbles(16);
  private readonly bubbleRig = new Group();
  private time = 0;

  constructor(geometry: BufferGeometry, coarse: boolean) {
    const { material, uniforms } = createDeformMaterial(
      {
        color: 0xffffff,
        metalness: 0,
        roughness: 0.03,
        transmission: 1,
        thickness: 1.7,
        ior: 1.333,
        attenuationColor: new Color(0.6, 0.86, 1.0),
        attenuationDistance: 2.4,
        specularIntensity: 1,
        envMapIntensity: 1.15,
        dispersion: coarse ? 0 : 0.12,
      },
      DeformMode.Water,
    );
    const mesh = new Mesh(geometry, material);
    mesh.frustumCulled = false;
    this.ripples = new RippleField(uniforms);
    this.bubbleRig.add(this.bubbles.mesh);
    this.objects = [mesh] as const;
    // Bubbles rise along world up, so their rig follows the anchor's position but not its spin.
    this.worldObjects = [this.bubbleRig] as const;
  }

  hint(): string {
    return 'Tap for ripples. Drag to stir the surface.';
  }

  readout(): Readout {
    return ['Water', 'n = 1.333', '997 kg/m³'];
  }

  down(dir: Vector3): void {
    this.ripples.down(dir, this.time);
  }
  move(dir: Vector3): void {
    this.ripples.move(dir, this.time);
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
    this.bubbleRig.position.copy(ctx.anchor.position);
    this.bubbleRig.scale.copy(ctx.anchor.scale);
    this.bubbles.update(ctx.time, ctx.dt);
  }
}
