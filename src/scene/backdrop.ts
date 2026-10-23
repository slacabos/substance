import {
  BackSide,
  Color,
  CustomBlending,
  Mesh,
  OneFactor,
  OneMinusSrcAlphaFactor,
  PlaneGeometry,
  SphereGeometry,
} from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { uniform } from 'three/tsl';
import { approach } from '../core/spring';
import { backdropColor, poolColor } from '../nodes/backdrop';
import { SETTINGS, type SettingId, type SettingLook } from './settings';

/** How quickly a new setting fades in (per second). */
const FADE_RATE = 7;

/**
 * A big inside-out sphere painted as a studio, plus the pool of light or shadow on the floor.
 * The sphere is opaque, so transmissive materials refract it. Switching settings fades every value across.
 */
export class Backdrop {
  readonly sphere: Mesh;
  readonly pool: Mesh;

  private readonly uniforms = {
    wall: uniform(new Color()),
    floor: uniform(new Color()),
    horizon: uniform(new Color()),
    halo: uniform(new Color()),
  };
  private readonly poolUniforms = {
    color: uniform(new Color()),
    strength: uniform(0.2),
    glow: uniform(1),
    shadow: uniform(0),
  };
  private target: SettingLook;

  constructor(setting: SettingId) {
    const sky = new MeshBasicNodeMaterial({ side: BackSide, depthWrite: false });
    sky.colorNode = backdropColor(this.uniforms);
    this.sphere = new Mesh(new SphereGeometry(40, 64, 32), sky);
    this.sphere.renderOrder = -1;

    const pool = new MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      toneMapped: false,
      blending: CustomBlending,
      blendSrc: OneFactor,
      blendDst: OneMinusSrcAlphaFactor,
    });
    pool.colorNode = poolColor(this.poolUniforms);
    this.pool = new Mesh(new PlaneGeometry(3.2, 3.2), pool);
    this.pool.rotation.x = -Math.PI / 2;
    this.pool.position.y = -1.55;

    this.target = SETTINGS[setting];
    this.update(1, true);
  }

  setSetting(id: SettingId, instant = false): void {
    this.target = SETTINGS[id];
    if (instant) this.update(1, true);
  }

  /** The coloured glow each material casts on the floor. */
  setPool(color: string, strength: number): void {
    this.poolUniforms.color.value.set(color);
    this.poolUniforms.strength.value = strength;
  }

  update(dt: number, instant = false): void {
    const k = instant ? 1 : approach(FADE_RATE, dt);
    const u = this.uniforms;
    const t = this.target;
    u.wall.value.lerp(t.wall, k);
    u.floor.value.lerp(t.floor, k);
    u.horizon.value.lerp(t.horizon, k);
    u.halo.value.lerp(t.halo, k);
    const p = this.poolUniforms;
    p.glow.value += (t.glow - p.glow.value) * k;
    p.shadow.value += (t.shadow - p.shadow.value) * k;
  }
}
