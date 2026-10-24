import type { Object3D, Vector3 } from 'three';
import type { SettingLook } from '../scene/settings';

export type MaterialId = 'water' | 'ocean' | 'coast' | 'slime' | 'fur';
export type ShapeId = 'orb' | 'cube';

export interface FrameContext {
  time: number;
  dt: number;
  /** The rotating object everything is attached to. */
  anchor: Object3D;
}

/** The figures shown in the top-left readout: a title and two detail lines. */
export type Readout = readonly [title: string, line1: string, line2: string];

/**
 * One material the object can be made of. Directions passed in are unit vectors in object space,
 * pointing from the centre to where the finger or cursor touches.
 */
export interface Substance {
  readonly id: MaterialId;
  readonly accent: string;
  readonly poolStrength: number;
  /** How far beyond the base surface touches still count (fur sticks out). */
  readonly hitPadding: number;
  /** Attached to the spinning anchor; shown only while this material is active. */
  readonly objects: readonly Object3D[];
  /** Added to the scene directly (they don't spin); shown only while this material is active. */
  readonly worldObjects?: readonly Object3D[];

  hint(coarse: boolean): string;
  readout(): Readout;

  down(dir: Vector3): void;
  move(dir: Vector3, dt: number): void;
  up(): void;
  enter(front: Vector3): void;
  update(ctx: FrameContext): void;

  hover?(dir: Vector3, dt: number): void;
  hoverEnd?(): void;
  onShapeChange?(front: Vector3): void;
  onSpinJerk?(axisLocal: Vector3, magnitude: number): void;
  /** The backdrop changed; materials that reflect a sun follow it. */
  onSettingChange?(look: SettingLook): void;
}
