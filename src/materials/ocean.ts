import { Color, IcosahedronGeometry, Mesh, Vector3, type BufferGeometry, type Texture } from 'three';
import { MeshBasicNodeMaterial, type WebGPURenderer } from 'three/webgpu';
import { Fn, normalize, positionGeometry } from 'three/tsl';
import { TapRhythm } from '../core/oceanRhythm';
import { OceanWeather, weatherRadius } from '../core/oceanWeather';
import { beaufortFor, GustField, oceanWind } from '../core/oceanWind';
import type { OceanTuning } from '../core/oceanTuning';
import { baseR } from '../nodes/shape';
import { morphUniform } from '../scene/shape';
import { createDeformMaterial, createDeformUniforms, DeformMode, type DeformUniforms } from './deform';
import type { OceanFFT } from './oceanFFT';
import { OceanFoamFlecks } from './oceanFoamFlecks';
import { OceanSim, oceanReleaseImpulse, type OceanForce } from './oceanSim';
import { OceanShores } from './oceanShores';
import { OceanSpray } from './oceanSpray';
import type { SettingLook } from '../scene/settings';
import type { FrameContext, MaterialId, Readout, Substance } from './types';

/** A water-covered miniature orb with coherent swells and local, touch-driven rough weather. */
export class Ocean implements Substance {
  readonly id: MaterialId;
  readonly accent = '#328eaf';
  readonly hasWind = true;
  readonly poolStrength = 0.14;
  readonly hitPadding = 0.1;
  readonly objects;

  private readonly sim: OceanSim;
  private readonly waves: OceanFFT;
  private readonly shores: OceanShores | null;
  private readonly spray: OceanSpray;
  private readonly foamFlecks: OceanFoamFlecks;
  private readonly weatherField = new OceanWeather();
  private readonly gusts: GustField;
  private readonly rhythm = new TapRhythm();
  /** How much harder the current press pushes, for keeping the beat. */
  private boost = 1;
  private readonly uniforms: DeformUniforms;
  readonly tuning: OceanTuning;
  private readonly lastDrag = new Vector3(0, 0, 1);
  private readonly pressDir = new Vector3(0, 0, 1);
  private readonly pendingForces: OceanForce[] = [];
  private pressing = false;
  private heldSeconds = 0;
  private pressStartedAt = 0;
  /** The live wind plus touch-made rough weather, for the readout. */
  private beaufort = 2;
  private time = 0;
  private lastSprayTime = -Infinity;
  private lastWhitewaterTime = -Infinity;

  constructor(
    geometry: BufferGeometry,
    coarse: boolean,
    skyReflection: Texture,
    tuning: OceanTuning,
    waves: OceanFFT,
    renderer: WebGPURenderer,
    coastal = false,
  ) {
    this.id = coastal ? 'coast' : 'ocean';
    this.tuning = tuning;
    this.gusts = new GustField(coastal ? 2 : 1);
    this.waves = waves;
    this.shores = coastal ? new OceanShores(coarse) : null;
    const deformUniforms = createDeformUniforms();
    this.sim = new OceanSim(renderer, geometry, coastal, tuning, { fields: waves, uniforms: deformUniforms });
    const simBuffers = this.sim.renderBuffers();
    this.spray = new OceanSpray(coarse);
    const { material, uniforms } = createDeformMaterial(
      {
        color: 0xffffff,
        metalness: 0,
        roughness: 0.2,
        specularIntensity: 0.75,
        ior: 1.333,
        transmission: 0.22,
        thickness: 0.7,
        attenuationColor: new Color('#082f49'),
        attenuationDistance: 0.5,
        envMap: skyReflection,
        envMapIntensity: 0.92,
      },
      DeformMode.Ocean,
      { waves, sim: simBuffers, spacing: this.sim.spacing, coastal },
      deformUniforms,
    );
    this.foamFlecks = new OceanFoamFlecks(renderer, coarse, {
      buffers: simBuffers,
      waves,
      uniforms,
      spacing: this.sim.spacing,
      count: this.sim.count,
    });
    const mesh = new Mesh(geometry, material);
    mesh.frustumCulled = false;
    const interiorMaterial = new MeshBasicNodeMaterial({ color: '#031525' });
    interiorMaterial.positionNode = Fn(() => {
      const p = normalize(positionGeometry);
      return p.mul(baseR(p).mul(0.72));
    })();
    const interior = new Mesh(new IcosahedronGeometry(1, 24), interiorMaterial);
    interior.frustumCulled = false;
    this.uniforms = uniforms;
    this.applyTuning();
    this.objects = [
      mesh,
      interior,
      this.foamFlecks.mesh,
      this.spray.mesh,
      this.spray.mistMesh,
      ...(this.shores ? [this.shores.mesh] : []),
    ];
  }

  hint(): string {
    return this.shores
      ? 'Press for waves. Watch them break on the scattered shores.'
      : 'Press longer for bigger waves. Drag to churn the sea.';
  }

  readout(): Readout {
    return [this.shores ? 'Coast' : 'Ocean', `Beaufort ${this.beaufort}`, '35 g/kg salt'];
  }

  down(dir: Vector3): void {
    this.weatherField.down(dir);
    this.lastDrag.copy(dir);
    this.pressDir.copy(dir);
    this.pressing = true;
    this.heldSeconds = 0;
    this.pressStartedAt = performance.now();
    this.boost = this.rhythm.tap(this.pressStartedAt / 1000);
    this.sim.setPressure(dir, 0);
    this.addForce(dir, -0.55 * this.boost);
  }
  move(dir: Vector3, dt: number): void {
    this.weatherField.move(dir, dt);
    this.pressDir.copy(dir);
    if (this.pressing) this.sim.setPressure(dir, this.heldSeconds);
    const moved = this.lastDrag.angleTo(dir);
    if (moved > 0.015) {
      const strength = Math.min(0.5, (moved / Math.max(dt, 1 / 120)) * 0.14);
      const motion = dir.clone().sub(this.lastDrag).normalize();
      this.addForce(this.lastDrag, -strength, motion);
      this.addForce(dir, strength * 1.2, motion);
      if (moved > 0.04 && this.time - this.lastSprayTime > 0.12) {
        this.spray.splash(dir, Math.min(0.6, strength * 1.4), 5);
        this.lastSprayTime = this.time;
      }
      this.lastDrag.copy(dir);
    }
  }
  up(): void {
    this.weatherField.up();
    if (this.pressing) {
      this.heldSeconds = Math.min(2, (performance.now() - this.pressStartedAt) / 1000);
      this.addForce(this.pressDir, oceanReleaseImpulse(this.heldSeconds) * this.boost);
    }
    this.pressing = false;
    this.sim.setPressure(null, 0);
  }
  enter(front: Vector3): void {
    this.lastDrag.copy(front);
    this.pressing = false;
    this.heldSeconds = 0;
    this.sim.reset();
    this.rhythm.reset();
    this.spray.clear();
    this.foamFlecks.clear();
    this.weatherField.clear();
  }

  update(ctx: FrameContext): void {
    this.time = ctx.time;
    this.uniforms.uTime.value = ctx.time;
    if (this.objects[0]!.visible) {
      this.waves.update(ctx.time);
      this.shores?.updateMorph(morphUniform.value);
      if (this.pressing) {
        this.heldSeconds = Math.min(2, (performance.now() - this.pressStartedAt) / 1000);
        this.sim.setPressure(this.pressDir, this.heldSeconds);
      }
      this.weatherField.update(ctx.dt);
      this.sim.update(ctx.dt, this.pendingForces, ctx.time);
      if (ctx.time - this.lastWhitewaterTime > 0.075) {
        for (const site of this.sim.whitewaterSites(8)) {
          this.spray.emitCrest(
            site.dir,
            site.flow,
            site.strength,
            site.coastal ? 3 : 2,
            site.coastal ? site.height : undefined,
          );
        }
        this.lastWhitewaterTime = ctx.time;
      }
      this.spray.update(ctx.dt, this.sim);
      this.foamFlecks.update(ctx.dt);
    }
    this.pendingForces.length = 0;
    // the live wind (the same function the FFT reads, so they agree)
    const wind = oceanWind(ctx.time, this.tuning.windSpeed);
    this.sim.setWind(wind.angle, wind.speed);
    this.foamFlecks.setWindAngle(wind.angle);
    this.uniforms.uOceanWind.value = wind.speed;
    this.gusts.update(ctx.dt, wind);
    // cat's paws need wind to raise them
    const gustScale = Math.min(1, wind.speed / 0.6);
    this.gusts.gusts.forEach((gust, i) => {
      this.uniforms.gusts[i]!.set(
        gust.center.x,
        gust.center.y,
        gust.center.z,
        this.gusts.intensity(gust) * gustScale,
      );
    });
    this.beaufort = beaufortFor(wind.speed, this.weatherField.peak);
    for (let i = 0; i < this.weatherField.patches.length; i++) {
      const patch = this.weatherField.patches[i]!;
      this.uniforms.weather[i]!.set(patch.center.x, patch.center.y, patch.center.z, patch.strength);
      this.uniforms.weatherB[i]!.x = weatherRadius(patch.age);
    }
  }

  onSettingChange(look: SettingLook): void {
    this.uniforms.uSunDir.value.copy(look.sun.direction);
    this.uniforms.uSunColor.value.copy(look.sun.color);
  }

  private addForce(dir: Vector3, strength: number, motion?: Vector3): void {
    if (this.pendingForces.length === 4) this.pendingForces.shift();
    this.pendingForces.push(
      motion ? { dir: dir.clone(), strength, motion: motion.clone() } : { dir: dir.clone(), strength },
    );
  }

  private applyTuning(): void {
    this.uniforms.uOceanWavelength.value = this.tuning.peakWavelength;
    this.uniforms.uOceanAmplitude.value = this.tuning.amplitude;
    this.uniforms.uOceanChoppiness.value = this.tuning.choppiness;
    this.uniforms.uOceanFoam.value = this.tuning.foam;
    this.foamFlecks.setIntensity(this.tuning.foam);
  }
}
