import { Color, IcosahedronGeometry, Mesh, Vector3, type BufferGeometry, type Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { Fn, normalize, positionGeometry } from 'three/tsl';
import { OceanWeather, weatherRadius } from '../core/oceanWeather';
import type { OceanTuning } from '../core/oceanTuning';
import { baseR } from '../nodes/shape';
import { createDeformMaterial, DeformMode, type DeformUniforms } from './deform';
import { OceanFoamFlecks } from './oceanFoamFlecks';
import { OceanSim, oceanReleaseImpulse, type OceanForce } from './oceanSim';
import { OceanSpray } from './oceanSpray';
import type { FrameContext, Readout, Substance } from './types';

/** Storm level (1 = calm) to a Beaufort number for the readout. */
export function beaufortFor(storm: number): number {
  return Math.round(2 + (storm - 1) * 4);
}

/** A water-covered miniature orb with coherent swells and local, touch-driven rough weather. */
export class Ocean implements Substance {
  readonly id = 'ocean';
  readonly accent = '#328eaf';
  readonly poolStrength = 0.14;
  readonly hitPadding = 0.1;
  readonly objects;

  private readonly sim: OceanSim;
  private readonly spray: OceanSpray;
  private readonly foamFlecks: OceanFoamFlecks;
  private readonly weatherField = new OceanWeather();
  private readonly uniforms: DeformUniforms;
  readonly tuning: OceanTuning;
  private readonly lastDrag = new Vector3(0, 0, 1);
  private readonly pressDir = new Vector3(0, 0, 1);
  private readonly pendingForces: OceanForce[] = [];
  private pressing = false;
  private heldSeconds = 0;
  private pressStartedAt = 0;
  private storm = 1;
  private time = 0;
  private lastSprayTime = -Infinity;
  private lastWhitewaterTime = -Infinity;

  constructor(geometry: BufferGeometry, coarse: boolean, skyReflection: Texture, tuning: OceanTuning) {
    this.tuning = tuning;
    this.sim = new OceanSim(geometry);
    this.spray = new OceanSpray(coarse);
    this.foamFlecks = new OceanFoamFlecks(coarse, tuning);
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
    );
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
    this.objects = [mesh, interior, this.foamFlecks.mesh, this.spray.mesh, this.spray.mistMesh] as const;
  }

  hint(): string {
    return 'Press longer for bigger waves. Drag to churn the sea.';
  }

  readout(): Readout {
    return ['Ocean', `Beaufort ${beaufortFor(this.storm)}`, '35 g/kg salt'];
  }

  down(dir: Vector3): void {
    this.weatherField.down(dir);
    this.lastDrag.copy(dir);
    this.pressDir.copy(dir);
    this.pressing = true;
    this.heldSeconds = 0;
    this.pressStartedAt = performance.now();
    this.sim.setPressure(dir, 0);
    this.addForce(dir, -0.55);
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
      this.addForce(this.pressDir, oceanReleaseImpulse(this.heldSeconds));
    }
    this.pressing = false;
    this.sim.setPressure(null, 0);
  }
  enter(front: Vector3): void {
    this.lastDrag.copy(front);
    this.pressing = false;
    this.heldSeconds = 0;
    this.sim.reset();
    this.spray.clear();
    this.foamFlecks.clear();
    this.weatherField.clear();
  }

  update(ctx: FrameContext): void {
    this.time = ctx.time;
    this.uniforms.uTime.value = ctx.time;
    if (this.objects[0].visible) {
      if (this.pressing) {
        this.heldSeconds = Math.min(2, (performance.now() - this.pressStartedAt) / 1000);
        this.sim.setPressure(this.pressDir, this.heldSeconds);
      }
      this.weatherField.update(ctx.dt);
      this.sim.update(ctx.dt, this.pendingForces);
      if (ctx.time - this.lastWhitewaterTime > 0.075) {
        for (const site of this.sim.whitewaterSites(8)) {
          this.spray.emitCrest(site.dir, site.flow, site.strength, 2);
          this.foamFlecks.emit(site.dir, site.flow, site.strength, 4);
        }
        this.lastWhitewaterTime = ctx.time;
      }
      this.spray.update(ctx.dt, this.sim);
      this.foamFlecks.update(ctx.dt, ctx.time, this.sim, this.weatherField);
    }
    this.pendingForces.length = 0;
    this.storm = 1 + this.weatherField.peak * 1.4;
    for (let i = 0; i < this.weatherField.patches.length; i++) {
      const patch = this.weatherField.patches[i]!;
      this.uniforms.weather[i]!.set(patch.center.x, patch.center.y, patch.center.z, patch.strength);
      this.uniforms.weatherB[i]!.x = weatherRadius(patch.age);
    }
  }

  private addForce(dir: Vector3, strength: number, motion?: Vector3): void {
    if (this.pendingForces.length === 4) this.pendingForces.shift();
    this.pendingForces.push(
      motion ? { dir: dir.clone(), strength, motion: motion.clone() } : { dir: dir.clone(), strength },
    );
  }

  private applyTuning(): void {
    this.uniforms.uOceanWind.value = this.tuning.windSpeed;
    this.uniforms.uOceanWavelength.value = this.tuning.peakWavelength;
    this.uniforms.uOceanAmplitude.value = this.tuning.amplitude;
    this.uniforms.uOceanChoppiness.value = this.tuning.choppiness;
    this.uniforms.uOceanFoam.value = this.tuning.foam;
    this.foamFlecks.setIntensity(this.tuning.foam);
    this.sim.setWindSpeed(this.tuning.windSpeed);
  }
}
