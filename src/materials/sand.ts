import { DirectionalLight, Mesh, Vector3 } from 'three';
import { MeshStandardNodeMaterial, type WebGPURenderer } from 'three/webgpu';
import {
  float,
  Fn,
  length,
  normalize,
  normalLocal,
  positionGeometry,
  uniform,
  varyingProperty,
  vec3,
  vertexIndex,
} from 'three/tsl';
import { oceanDetail } from '../core/oceanTuning';
import { reposeSlope, SAND } from '../core/sand';
import { SAND_KEY_LIGHT, sandSurface } from '../nodes/sand';
import { baseN, baseR } from '../nodes/shape';
import { createDeformGeometry } from './deform';
import { SandField } from './sandField';
import { SandGrains } from './sandGrains';
import type { SettingLook } from '../scene/settings';
import type { FrameContext, Readout, Substance } from './types';

/** How bright the sand's key light (SAND_KEY_LIGHT) is in its strongest channel. */
const KEY_STRENGTH = 2.4;

/** How the hand works sand. */
const HAND = {
  /** A tap's crater: depth at the centre and footprint sharpness (about 0.1 radians across). */
  tap: { depth: 0.012, sharpness: 140 },
  /** Holding presses deeper, this much a second, up to `most`, through a smaller, sharper footprint. */
  press: { rate: 0.03, most: 0.03, sharpness: 220, after: 0.2 },
  /** A drag ploughs a ridge this high per radian per second of hand speed, at most `most`. */
  plough: { gain: 0.006, most: 0.008, sharpness: 160 },
  /** A drag faster than this, radians per second, throws grains (all of a throw's at twice it). */
  flick: { speed: 2.5 },
} as const;

/**
 * An orb of sand: a sand bed over the shape, dug, ploughed and pressed by the hand, which slides back to its angle
 * of repose (materials/sandField.ts on core/sand.ts). Spinning it hard shakes loose its steepest slopes.
 */
export class Sand implements Substance {
  readonly id = 'sand';
  readonly accent = '#d9b77e';
  readonly poolStrength = 0.12;
  readonly hitPadding = 0;
  readonly objects;
  /**
   * The setting's sun, raking across the sand: without it the soft room light shows a dug crater or a furrow
   * only as a faint change of tone. Only lit while the sand shows.
   */
  readonly worldObjects;
  private readonly sun = new DirectionalLight(0xffffff, 0);
  /** The key light's colour, for the grains' glints. */
  private readonly sunColor = uniform(new Vector3());

  private readonly field: SandField;
  private readonly grains: SandGrains;
  private readonly lastDir = new Vector3(0, 0, 1);
  private readonly pressDir = new Vector3(0, 0, 1);
  private readonly motion = new Vector3();
  private pressing = false;
  private held = 0;
  private pressed = 0;

  constructor(renderer: WebGPURenderer, coarse: boolean) {
    const geometry = createDeformGeometry(true, oceanDetail(coarse));
    this.field = new SandField(renderer, geometry);
    this.grains = new SandGrains(renderer, coarse, this.field);
    const { bed, slope } = this.field.renderBuffers();
    const wet = varyingProperty('float', 'vWet');
    const disturbed = varyingProperty('float', 'vDisturbed');
    const below = varyingProperty('float', 'vBelow');
    const steep = varyingProperty('float', 'vSteep');
    const down = varyingProperty('vec3', 'vDown');
    const material = new MeshStandardNodeMaterial();
    material.positionNode = Fn(() => {
      const p = normalize(positionGeometry);
      const here = bed.element(vertexIndex);
      const radius = baseR(p);
      wet.assign(here.y);
      disturbed.assign(here.z);
      below.assign(here.x.negate());
      // the shape's normal, tilted by the bed's slope
      const gradient = slope.element(vertexIndex).xyz;
      normalLocal.assign(normalize(baseN(p, p.mul(radius)).sub(gradient.div(radius))));
      steep.assign(length(gradient).div(reposeSlope(0)));
      down.assign(gradient.negate().div(length(gradient).add(1e-6)));
      return p.mul(radius.add(here.x));
    })();
    const surface = sandSurface(
      normalize(positionGeometry),
      wet,
      disturbed,
      { dir: vec3(SAND_KEY_LIGHT.x, SAND_KEY_LIGHT.y, SAND_KEY_LIGHT.z), color: this.sunColor },
      below,
      { steep, down },
    );
    material.colorNode = surface.color;
    material.roughnessNode = surface.roughness;
    material.normalNode = surface.normal;
    material.metalnessNode = float(0);
    material.emissiveNode = surface.emissive;
    // the room's soft light at reduced strength, so the sun's raking light shows the relief
    material.envMapIntensity = 0.6;
    const mesh = new Mesh(geometry, material);
    mesh.frustumCulled = false;
    this.objects = [mesh, this.grains.mesh] as const;
    this.worldObjects = [this.sun] as const;
  }

  hint(): string {
    return 'Tap to dig. Hold for a footprint. Drag to plough, flick to throw sand.';
  }

  readout(): Readout {
    return ['Sand', `repose ${SAND.repose.dry}°`, 'grain 0.3 mm'];
  }

  down(dir: Vector3): void {
    this.lastDir.copy(dir);
    this.pressDir.copy(dir);
    this.pressing = true;
    this.held = 0;
    this.pressed = 0;
    this.field.add({ dir: [dir.x, dir.y, dir.z], amount: HAND.tap.depth, sharpness: HAND.tap.sharpness });
  }

  move(dir: Vector3, dt: number): void {
    const moved = this.lastDir.angleTo(dir);
    if (moved < 0.006) return;
    this.pressDir.copy(dir);
    // a moving hand doesn't press a footprint
    this.held = 0;
    const speed = moved / Math.max(dt, 1 / 120);
    this.motion.copy(dir).sub(this.lastDir);
    this.motion.addScaledVector(dir, -this.motion.dot(dir)).normalize();
    const amount = Math.min(HAND.plough.most, speed * HAND.plough.gain);
    this.field.add({
      dir: [dir.x, dir.y, dir.z],
      amount,
      sharpness: HAND.plough.sharpness,
      motion: [this.motion.x, this.motion.y, this.motion.z],
    });
    // a flick throws grains along it
    if (speed > HAND.flick.speed) {
      const share = Math.min(1, (speed - HAND.flick.speed) / HAND.flick.speed);
      this.grains.throw(
        [dir.x, dir.y, dir.z],
        [this.motion.x, this.motion.y, this.motion.z],
        speed * 0.25,
        share,
      );
    }
    this.lastDir.copy(dir);
  }

  up(): void {
    this.pressing = false;
  }

  enter(front: Vector3): void {
    this.lastDir.copy(front);
    this.pressing = false;
  }

  onSettingChange(look: SettingLook): void {
    // a key light from the upper left front, tinted like the setting's sun but at a steady strength: the sun
    // itself is set for water's glints and often shines from behind, where it can't rake across the near side
    this.sun.position.copy(SAND_KEY_LIGHT).multiplyScalar(10);
    const { r, g, b } = look.sun.color;
    this.sun.color.setRGB(r, g, b).multiplyScalar(KEY_STRENGTH / Math.max(r, g, b, 1e-3));
    this.sun.intensity = 1;
    this.sunColor.value.set(this.sun.color.r, this.sun.color.g, this.sun.color.b);
  }

  onSpinJerk(_axis: Vector3, magnitude: number): void {
    this.field.setShake(Math.min(1, magnitude * 0.4));
  }

  update(ctx: FrameContext): void {
    if (!this.objects[0].visible) return;
    if (this.pressing) {
      this.held += ctx.dt;
      // a held press sinks a deeper footprint, until it's as deep as it goes
      if (this.held > HAND.press.after && this.pressed < HAND.press.most) {
        const depth = Math.min(HAND.press.rate * ctx.dt, HAND.press.most - this.pressed);
        this.pressed += depth;
        const d = this.pressDir;
        this.field.add({ dir: [d.x, d.y, d.z], amount: depth, sharpness: HAND.press.sharpness });
      }
    }
    this.grains.update(ctx.dt);
    this.field.update(ctx.dt);
  }
}
