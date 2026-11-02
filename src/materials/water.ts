import { Color, Group, IcosahedronGeometry, Mesh, type BufferGeometry, type Vector3 } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { Fn, normalLocal, normalize, positionGeometry } from 'three/tsl';
import { shapeNormal, shapeShell, shapeMask } from '../nodes/shape';
import { UNFURL } from '../core/transition';
import { planeUniform } from '../scene/shape';
import { RisingBubbles } from './bubbles';
import { createDeformMaterial, DeformMode } from './deform';
import { RippleField } from './ripples';
import type { FrameContext, Readout, Substance } from './types';

/** How deep the plane's water stands over its bed, as a fraction of the orb's radius. */
const BED_DEPTH = 0.85;

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
  /** A pale floor under the plane's water, for it to refract; inside the orb there's nothing to see. */
  private readonly bed: Mesh;
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
    const bedMaterial = new MeshStandardNodeMaterial({ color: '#b9cfd6', roughness: 0.9 });
    bedMaterial.maskNode = shapeMask(normalize(positionGeometry));
    bedMaterial.positionNode = Fn(() => {
      const p = normalize(positionGeometry);
      normalLocal.assign(shapeNormal(p, p));
      return shapeShell(p.mul(BED_DEPTH), p);
    })();
    this.bed = new Mesh(new IcosahedronGeometry(1, 24), bedMaterial);
    this.bed.frustumCulled = false;
    // (in a rig of its own: the app shows and hides the material's objects, the bed only shows on the plane)
    const bedRig = new Group().add(this.bed);
    this.objects = [mesh, bedRig] as const;
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
    // (a flat sheet of water has nothing to rise through)
    this.bubbles.mesh.visible = planeUniform.value < UNFURL.bubblesUntil;
    this.bed.visible = planeUniform.value > 0;
    this.bubbles.update(ctx.time, ctx.dt);
  }
}
