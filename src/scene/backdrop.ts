import {
  BackSide,
  Color,
  CustomBlending,
  Mesh,
  OneFactor,
  OneMinusSrcAlphaFactor,
  PlaneGeometry,
  ShaderMaterial,
  SphereGeometry,
} from 'three';
import { approach } from '../core/spring';
import backdropFrag from '../shaders/backdrop.frag.glsl?raw';
import backdropVert from '../shaders/backdrop.vert.glsl?raw';
import poolFrag from '../shaders/pool.frag.glsl?raw';
import poolVert from '../shaders/pool.vert.glsl?raw';
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
    uWall: { value: new Color() },
    uFloor: { value: new Color() },
    uHorizon: { value: new Color() },
    uHalo: { value: new Color() },
  };
  private readonly poolUniforms = {
    uColor: { value: new Color() },
    uA: { value: 0.2 },
    uGlow: { value: 1 },
    uShadow: { value: 0 },
  };
  private target: SettingLook;

  constructor(setting: SettingId) {
    this.sphere = new Mesh(
      new SphereGeometry(40, 64, 32),
      new ShaderMaterial({
        side: BackSide,
        depthWrite: false,
        uniforms: this.uniforms,
        vertexShader: backdropVert,
        fragmentShader: backdropFrag,
      }),
    );
    this.sphere.renderOrder = -1;

    this.pool = new Mesh(
      new PlaneGeometry(3.2, 3.2),
      new ShaderMaterial({
        transparent: true,
        depthWrite: false,
        blending: CustomBlending,
        blendSrc: OneFactor,
        blendDst: OneMinusSrcAlphaFactor,
        uniforms: this.poolUniforms,
        vertexShader: poolVert,
        fragmentShader: poolFrag,
      }),
    );
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
    this.poolUniforms.uColor.value.set(color);
    this.poolUniforms.uA.value = strength;
  }

  update(dt: number, instant = false): void {
    const k = instant ? 1 : approach(FADE_RATE, dt);
    const u = this.uniforms;
    const t = this.target;
    u.uWall.value.lerp(t.wall, k);
    u.uFloor.value.lerp(t.floor, k);
    u.uHorizon.value.lerp(t.horizon, k);
    u.uHalo.value.lerp(t.halo, k);
    const p = this.poolUniforms;
    p.uGlow.value += (t.glow - p.uGlow.value) * k;
    p.uShadow.value += (t.shadow - p.uShadow.value) * k;
  }
}
