import { BufferAttribute, DynamicDrawUsage, Vector3, type BufferGeometry } from 'three';
import { oceanMesh } from '../core/oceanMesh';

export interface OceanForce {
  dir: Vector3;
  strength: number;
  motion?: Vector3;
}

export interface OceanWhitewaterSite {
  dir: Vector3;
  flow: Vector3;
  strength: number;
}

export interface OceanSample {
  height: number;
  velocity: number;
  foam: number;
  flowX: number;
  flowY: number;
  flowZ: number;
}

const smoothstep = (a: number, b: number, x: number): number => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const fract = (x: number): number => x - Math.floor(x);
const CELL = 0.04;
const GRID = 64;
const cell = (v: number): number => Math.max(0, Math.min(GRID - 1, Math.floor((v + 1.2) / CELL)));
const key = (x: number, y: number, z: number): number => x * GRID * GRID + y * GRID + z;

/** A spherical free-surface wave field with tangential flow and advected whitewater. */
export class OceanSim {
  readonly count: number;
  private readonly positions: Float32Array;
  private readonly neighbors: Uint32Array;
  private readonly forceGrain: Float32Array;
  private readonly foamGrain: Float32Array;
  private readonly force: Float32Array;
  private readonly flowImpulse: Float32Array;
  private readonly flow: Float32Array;
  private readonly baseFlow: Float32Array;
  private readonly drift: Float32Array;
  private readonly spatial = new Map<number, number[]>();
  private readonly stiffness: number;
  private readonly invSpacing: number;
  private height: Float32Array;
  private nextHeight: Float32Array;
  private velocity: Float32Array;
  private nextVelocity: Float32Array;
  private foam: Float32Array;
  private nextFoam: Float32Array;
  private turbulence: Float32Array;
  private nextTurbulence: Float32Array;
  private readonly heightUpload: Float32Array;
  private readonly foamUpload: Float32Array;
  private readonly turbulenceUpload: Float32Array;
  private readonly gradientUpload: Float32Array;
  private readonly driftAttribute: BufferAttribute;
  private readonly heightAttribute: BufferAttribute;
  private readonly foamAttribute: BufferAttribute;
  private readonly turbulenceAttribute: BufferAttribute;
  private readonly gradientAttribute: BufferAttribute;
  private windSpeed = 1;

  constructor(geometry: BufferGeometry) {
    const mesh = oceanMesh(geometry);
    this.positions = mesh.positions;
    this.neighbors = mesh.neighbors;
    this.count = mesh.positions.length / 3;
    this.invSpacing = 1 / mesh.spacing;
    this.stiffness = (0.24 * 0.24 * 4) / (mesh.spacing * mesh.spacing);
    this.forceGrain = new Float32Array(this.count);
    this.foamGrain = new Float32Array(this.count);
    this.force = new Float32Array(this.count);
    this.flowImpulse = new Float32Array(this.count * 3);
    this.flow = new Float32Array(this.count * 3);
    this.baseFlow = new Float32Array(this.count * 3);
    this.drift = new Float32Array(this.count * 3);
    this.height = new Float32Array(this.count);
    this.nextHeight = new Float32Array(this.count);
    this.velocity = new Float32Array(this.count);
    this.nextVelocity = new Float32Array(this.count);
    this.foam = new Float32Array(this.count);
    this.nextFoam = new Float32Array(this.count);
    this.turbulence = new Float32Array(this.count);
    this.nextTurbulence = new Float32Array(this.count);
    this.heightUpload = new Float32Array(this.count);
    this.foamUpload = new Float32Array(this.count);
    this.turbulenceUpload = new Float32Array(this.count);
    this.gradientUpload = new Float32Array(this.count * 3);
    this.heightAttribute = new BufferAttribute(this.heightUpload, 1).setUsage(DynamicDrawUsage);
    this.foamAttribute = new BufferAttribute(this.foamUpload, 1).setUsage(DynamicDrawUsage);
    this.turbulenceAttribute = new BufferAttribute(this.turbulenceUpload, 1).setUsage(DynamicDrawUsage);
    this.gradientAttribute = new BufferAttribute(this.gradientUpload, 3).setUsage(DynamicDrawUsage);
    this.driftAttribute = new BufferAttribute(this.drift, 3).setUsage(DynamicDrawUsage);
    geometry.setAttribute('oceanHeight', this.heightAttribute);
    geometry.setAttribute('oceanFoam', this.foamAttribute);
    geometry.setAttribute('oceanTurbulence', this.turbulenceAttribute);
    geometry.setAttribute('oceanGradient', this.gradientAttribute);
    geometry.setAttribute('oceanDrift', this.driftAttribute);
    this.prepareStaticFields();
    for (let i = 0; i < this.count; i++) {
      const k = key(
        cell(this.positions[i * 3]!),
        cell(this.positions[i * 3 + 1]!),
        cell(this.positions[i * 3 + 2]!),
      );
      let bucket = this.spatial.get(k);
      if (!bucket) {
        bucket = [];
        this.spatial.set(k, bucket);
      }
      bucket.push(i);
    }
  }

  /** Interpolated nearby mesh state for whitewater and particle emission. */
  sampleAt(dir: Vector3): OceanSample {
    const x = cell(dir.x);
    const y = cell(dir.y);
    const z = cell(dir.z);
    let h = 0;
    let v = 0;
    let f = 0;
    let flowX = 0;
    let flowY = 0;
    let flowZ = 0;
    let weight = 0;
    this.forCells(x, y, z, 1, (i) => {
      const dx = this.positions[i * 3]! - dir.x;
      const dy = this.positions[i * 3 + 1]! - dir.y;
      const dz = this.positions[i * 3 + 2]! - dir.z;
      const w = 1 / (dx * dx + dy * dy + dz * dz + 0.0002);
      h += this.height[i]! * w;
      v += this.velocity[i]! * w;
      f += this.foam[i]! * w;
      flowX += this.flow[i * 3]! * w;
      flowY += this.flow[i * 3 + 1]! * w;
      flowZ += this.flow[i * 3 + 2]! * w;
      weight += w;
    });
    if (weight === 0) return { height: 0, velocity: 0, foam: 0, flowX: 0, flowY: 0, flowZ: 0 };
    return {
      height: h / weight,
      velocity: v / weight,
      foam: f / weight,
      flowX: flowX / weight,
      flowY: flowY / weight,
      flowZ: flowZ / weight,
    };
  }

  /** Small foam deposits made by splashes and falling spray. */
  depositFoam(dir: Vector3, amount: number, radius = 0.055): void {
    const x = cell(dir.x);
    const y = cell(dir.y);
    const z = cell(dir.z);
    const reach = Math.ceil(radius / CELL) + 1;
    const r2 = radius * radius;
    this.forCells(x, y, z, reach, (i) => {
      const dx = this.positions[i * 3]! - dir.x;
      const dy = this.positions[i * 3 + 1]! - dir.y;
      const dz = this.positions[i * 3 + 2]! - dir.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 >= r2) return;
      const falloff = 1 - smoothstep(0, r2, d2);
      this.foam[i] = clamp(this.foam[i]! + amount * falloff, 0, 1);
      this.turbulence[i] = clamp(this.turbulence[i]! + amount * falloff * 0.35, 0, 1);
    });
  }

  reset(): void {
    for (const field of [
      this.height,
      this.nextHeight,
      this.velocity,
      this.nextVelocity,
      this.foam,
      this.nextFoam,
      this.turbulence,
      this.nextTurbulence,
    ])
      field.fill(0);
    this.flow.set(this.baseFlow);
    this.flowImpulse.fill(0);
    this.gradientUpload.fill(0);
    this.drift.fill(0);
    this.upload();
  }

  setWindSpeed(speed: number): void {
    if (speed === this.windSpeed) return;
    this.windSpeed = speed;
    const p = this.positions;
    for (let i = 0; i < this.count; i++) {
      const x = p[i * 3]!;
      const y = p[i * 3 + 1]!;
      const z = p[i * 3 + 2]!;
      const baselineX = (z - y * 0.2) * 0.035 * speed;
      const baselineY = (x * 0.2 - z * 0.15) * 0.035 * speed;
      const baselineZ = (y * 0.15 - x) * 0.035 * speed;
      for (let axis = 0; axis < 3; axis++) {
        const j = i * 3 + axis;
        const next = axis === 0 ? baselineX : axis === 1 ? baselineY : baselineZ;
        this.flow[j] = this.flow[j]! + next - this.baseFlow[j]!;
        this.baseFlow[j] = next;
      }
    }
  }

  update(dt: number, events: readonly OceanForce[]): void {
    if (dt <= 0) return;
    const pairs = Math.min(4, Math.max(1, Math.ceil(dt * 60)));
    const passes = pairs * 2;
    const step = dt / passes;
    const impulseShare = 1 / passes;
    this.prepareForces(events);

    const velocityDrag = Math.exp(-step * 0.55);
    const foamDecay = Math.exp(-step * 0.8);
    const turbulenceDecay = Math.exp(-step * 0.9);
    const p = this.positions;
    const flow = this.flow;
    for (let pass = 0; pass < passes; pass++) {
      const h = this.height;
      const v = this.velocity;
      const f = this.foam;
      const t = this.turbulence;
      const nextH = this.nextHeight;
      const nextV = this.nextVelocity;
      const nextF = this.nextFoam;
      const nextT = this.nextTurbulence;
      for (let i = 0; i < this.count; i++) {
        const base = i * 6;
        const x = p[i * 3]!;
        const y = p[i * 3 + 1]!;
        const z = p[i * 3 + 2]!;
        const fx = flow[i * 3]!;
        const fy = flow[i * 3 + 1]!;
        const fz = flow[i * 3 + 2]!;
        let sumHeight = 0;
        let sumFoam = 0;
        let sumDelta = 0;
        let upwindFoam = 0;
        let upwindWeight = 0;
        let divergence = 0;
        for (let k = 0; k < 6; k++) {
          const j = this.neighbors[base + k]!;
          sumHeight += h[j]!;
          sumFoam += f[j]!;
          sumDelta += Math.abs(h[j]! - h[i]!);
          const dx = x - p[j * 3]!;
          const dy = y - p[j * 3 + 1]!;
          const dz = z - p[j * 3 + 2]!;
          const d2 = Math.max(1e-5, dx * dx + dy * dy + dz * dz);
          const weight = Math.max(0, dx * fx + dy * fy + dz * fz);
          upwindFoam += f[j]! * weight;
          upwindWeight += weight;
          divergence +=
            ((flow[j * 3]! - fx) * -dx + (flow[j * 3 + 1]! - fy) * -dy + (flow[j * 3 + 2]! - fz) * -dz) /
            d2 /
            6;
        }
        const slope = (sumDelta / 6) * this.invSpacing;
        const impulse = this.force[i]! * impulseShare;
        const velocity =
          (v[i]! + ((sumHeight / 6 - h[i]!) * this.stiffness - h[i]! * 0.9) * step + impulse) * velocityDrag;
        const height = clamp(h[i]! + velocity * step - divergence * step * 0.012, -0.065, 0.065);
        const breaking =
          smoothstep(0.09, 0.22, slope) *
          smoothstep(0.001, 0.015, height) *
          smoothstep(0.3, 0.8, this.foamGrain[i]!);
        const travel = clamp(step * Math.hypot(fx, fy, fz) * this.invSpacing * 0.65, 0, 0.35);
        const carried = f[i]! * (1 - travel) + (upwindFoam / Math.max(upwindWeight, 1e-5)) * travel;
        const foam = clamp(
          carried * foamDecay +
            (sumFoam / 6 - f[i]!) * step * 0.25 +
            breaking * step * 3 +
            Math.abs(impulse) * (0.35 + this.foamGrain[i]! * 0.65) * 0.55,
          0,
          1,
        );
        nextH[i] = height;
        nextV[i] = velocity;
        nextF[i] = foam;
        nextT[i] = clamp(
          Math.max(
            t[i]! * turbulenceDecay,
            slope * 0.9 + Math.abs(velocity) * 0.15 + Math.abs(divergence) * 0.02,
          ),
          0,
          1,
        );
      }
      this.height = nextH;
      this.nextHeight = h;
      this.velocity = nextV;
      this.nextVelocity = v;
      this.foam = nextF;
      this.nextFoam = f;
      this.turbulence = nextT;
      this.nextTurbulence = t;
    }
    this.updateFlow(dt);
    this.upload();
  }

  /** Sparse sites where a moving, aerated surface can shed spray. */
  whitewaterSites(maxCount: number): OceanWhitewaterSite[] {
    const sites: OceanWhitewaterSite[] = [];
    for (let i = 0; i < this.count; i++) {
      const g = i * 3;
      const slope = Math.hypot(
        this.gradientUpload[g]!,
        this.gradientUpload[g + 1]!,
        this.gradientUpload[g + 2]!,
      );
      const score = this.foam[i]! * 0.82 + this.turbulence[i]! * 0.3 + slope * 0.12;
      if (score < 0.18) continue;
      const dir = new Vector3(this.positions[g], this.positions[g + 1], this.positions[g + 2]);
      if (sites.some((site) => site.dir.dot(dir) > 0.9975)) continue;
      const flow = new Vector3(this.flow[g], this.flow[g + 1], this.flow[g + 2]);
      sites.push({ dir, flow, strength: clamp(score, 0.25, 0.9) });
      if (sites.length >= maxCount) break;
    }
    return sites;
  }

  private prepareStaticFields(): void {
    const p = this.positions;
    for (let i = 0; i < this.count; i++) {
      const x = p[i * 3]!;
      const y = p[i * 3 + 1]!;
      const z = p[i * 3 + 2]!;
      this.forceGrain[i] = 0.85 + Math.sin((x * 37 + y * 51 + z * 83) * 11) * 0.15;
      this.foamGrain[i] = fract(Math.sin((x * 43.7 + y * 91.3 + z * 67.9) * 67.3) * 43758.5453);
      this.baseFlow[i * 3] = (z - y * 0.2) * 0.035;
      this.baseFlow[i * 3 + 1] = (x * 0.2 - z * 0.15) * 0.035;
      this.baseFlow[i * 3 + 2] = (y * 0.15 - x) * 0.035;
    }
    this.flow.set(this.baseFlow);
  }

  private prepareForces(events: readonly OceanForce[]): void {
    this.force.fill(0);
    this.flowImpulse.fill(0);
    const p = this.positions;
    for (const event of events) {
      for (let i = 0; i < this.count; i++) {
        const alignment = p[i * 3]! * event.dir.x + p[i * 3 + 1]! * event.dir.y + p[i * 3 + 2]! * event.dir.z;
        if (alignment < 0.88) continue;
        const falloff = Math.exp((alignment - 1) * 110) * this.forceGrain[i]!;
        this.force[i] = this.force[i]! + event.strength * falloff;
        const x = p[i * 3]!;
        const y = p[i * 3 + 1]!;
        const z = p[i * 3 + 2]!;
        const dx = x - event.dir.x * alignment;
        const dy = y - event.dir.y * alignment;
        const dz = z - event.dir.z * alignment;
        const length = Math.hypot(dx, dy, dz) + 1e-5;
        const gain = Math.abs(event.strength) * falloff;
        const swirlX = event.dir.y * z - event.dir.z * y;
        const swirlY = event.dir.z * x - event.dir.x * z;
        const swirlZ = event.dir.x * y - event.dir.y * x;
        this.flowImpulse[i * 3] =
          this.flowImpulse[i * 3]! +
          (dx / length) * gain * 0.055 +
          (event.motion?.x ?? swirlX) * gain * 0.025;
        this.flowImpulse[i * 3 + 1] =
          this.flowImpulse[i * 3 + 1]! +
          (dy / length) * gain * 0.055 +
          (event.motion?.y ?? swirlY) * gain * 0.025;
        this.flowImpulse[i * 3 + 2] =
          this.flowImpulse[i * 3 + 2]! +
          (dz / length) * gain * 0.055 +
          (event.motion?.z ?? swirlZ) * gain * 0.025;
      }
    }
  }

  private updateFlow(dt: number): void {
    const drag = Math.exp(-dt * 0.75);
    const p = this.positions;
    for (let i = 0; i < this.count; i++) {
      const g = i * 3;
      const x = p[g]!;
      const y = p[g + 1]!;
      const z = p[g + 2]!;
      let gx = 0;
      let gy = 0;
      let gz = 0;
      for (let k = 0; k < 6; k++) {
        const j = this.neighbors[i * 6 + k]!;
        const nx = p[j * 3]!;
        const ny = p[j * 3 + 1]!;
        const nz = p[j * 3 + 2]!;
        const projection = x * nx + y * ny + z * nz;
        const tx = nx - x * projection;
        const ty = ny - y * projection;
        const tz = nz - z * projection;
        const dh = this.height[j]! - this.height[i]!;
        const factor = dh / Math.max(tx * tx + ty * ty + tz * tz, 1e-5) / 3;
        gx += tx * factor;
        gy += ty * factor;
        gz += tz * factor;
      }
      this.gradientUpload[g] = gx;
      this.gradientUpload[g + 1] = gy;
      this.gradientUpload[g + 2] = gz;
      const fx =
        this.flow[g]! * drag + this.baseFlow[g]! * (1 - drag) - gx * dt * 0.45 + this.flowImpulse[g]!;
      const fy =
        this.flow[g + 1]! * drag +
        this.baseFlow[g + 1]! * (1 - drag) -
        gy * dt * 0.45 +
        this.flowImpulse[g + 1]!;
      const fz =
        this.flow[g + 2]! * drag +
        this.baseFlow[g + 2]! * (1 - drag) -
        gz * dt * 0.45 +
        this.flowImpulse[g + 2]!;
      const radial = fx * x + fy * y + fz * z;
      this.flow[g] = clamp(fx - x * radial, -0.28, 0.28);
      this.flow[g + 1] = clamp(fy - y * radial, -0.28, 0.28);
      this.flow[g + 2] = clamp(fz - z * radial, -0.28, 0.28);
      this.drift[g] = this.drift[g]! + this.flow[g]! * dt;
      this.drift[g + 1] = this.drift[g + 1]! + this.flow[g + 1]! * dt;
      this.drift[g + 2] = this.drift[g + 2]! + this.flow[g + 2]! * dt;
    }
  }

  private upload(): void {
    this.heightUpload.set(this.height);
    this.foamUpload.set(this.foam);
    this.turbulenceUpload.set(this.turbulence);
    this.heightAttribute.needsUpdate = true;
    this.foamAttribute.needsUpdate = true;
    this.turbulenceAttribute.needsUpdate = true;
    this.gradientAttribute.needsUpdate = true;
    this.driftAttribute.needsUpdate = true;
  }

  private forCells(x: number, y: number, z: number, radius: number, visit: (i: number) => void): void {
    for (let a = Math.max(0, x - radius); a <= Math.min(GRID - 1, x + radius); a++)
      for (let b = Math.max(0, y - radius); b <= Math.min(GRID - 1, y + radius); b++)
        for (let c = Math.max(0, z - radius); c <= Math.min(GRID - 1, z + radius); c++) {
          const bucket = this.spatial.get(key(a, b, c));
          if (bucket) for (const i of bucket) visit(i);
        }
  }
}
