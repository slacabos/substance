import { DoubleSide, InstancedBufferGeometry, Mesh, PlaneGeometry, Vector3, Vector4 } from 'three';
import {
  MeshBasicNodeMaterial,
  type ComputeNode,
  type Node,
  type StorageBufferNode,
  type UniformArrayNode,
  type WebGPURenderer,
} from 'three/webgpu';
import {
  atomicAdd,
  atomicLoad,
  atomicStore,
  cameraProjectionMatrix,
  clamp,
  cross,
  Discard,
  dot,
  exp,
  float,
  floor,
  Fn,
  hash,
  If,
  instancedArray,
  instanceIndex,
  int,
  length,
  Loop,
  max,
  mix,
  modelViewMatrix,
  normalize,
  positionGeometry,
  Return,
  round,
  select,
  smoothstep,
  storage,
  uint,
  uniform,
  uniformArray,
  varyingProperty,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { OCEAN_FLUID } from '../config';
import { FLUID, fluidGrid } from '../core/oceanFluid';
import type { OceanField } from '../core/oceanField';
import { LOOKUP_SIZE } from '../core/oceanLookup';
import { shoreGroundHeight } from '../core/oceanShores';
import { oceanWeatherAt, type DeformUniforms } from '../nodes/deform';
import { cubeCell, type SurfaceBuffers } from '../nodes/oceanSurface';
import { oceanWaveVertex, type OceanWaveFields } from '../nodes/oceanWaves';
import { baseR } from '../nodes/shape';
import { FLUID_LAYER } from '../scene/fluidPass';

type Vec4Buffer = StorageBufferNode<'vec4'>;
type Vec3 = Node<'vec3'>;
type Float = Node<'float'>;

/** What the fluid needs of the sea it splashes from and lands on. */
export interface FluidSea {
  buffers: SurfaceBuffers;
  /** OceanSim.landings: water and foam the fluid takes from or gives back to each vertex. */
  landings: StorageBufferNode<'int'>;
  field: OceanField;
  waves: OceanWaveFields;
  uniforms: DeformUniforms;
  coastal: boolean;
}

/** Drag splashes waiting for the next update, and how many particles each throws. */
const SPLASHES = 4;
const PER_SPLASH = 24;
/** Particles started per second per unit of sphere area where the water breaks hardest, for 32k particles. */
const EMISSION = 30000;
/** How much harder breaking surf throws spray than open-sea whitecaps of the same strength. */
const SURF_SPRAY = 3;
/** A splash is a third droplets, the rest mist (as the old spray). */
const DROPLET_SHARE = 0.35;
/** A particle's crowd (see `crowd`) above which it's dense spray, drawn as one surface. */
const DENSE = 1;
/** Radius of a dense droplet's sphere in the fluid layer: about the spacing particles start at. */
const IMPOSTOR_RADIUS = 0.012;
/** Seconds of motion a streak shows. */
const STREAK_TIME = 0.04;
/** How solid a droplet's and a speck of mist's streak are at full life. */
const DROPLET_OPACITY = 0.7;
const MIST_OPACITY = 0.28;
/** Foam a landing leaves on its vertex. */
const LANDING_FOAM = { droplet: 0.02, mist: 0.004 };

/**
 * Splashes as a particle fluid on the GPU: the MLS-MPM step in core/oceanFluid.ts, with particles of mass 1
 * (the reference's volume divides out). Droplets are water that pushes against itself; mist rides ballistic in
 * the same pool. Particles start where the sea breaks (one pass over its vertices, claiming the oldest slots
 * of a ring) or under a drag, taking a little water from the vertex they leave; when they come down onto the
 * water they give it back with some foam, so landings leave small rings. All in the anchor's frame.
 */
export class OceanFluid {
  /** Isolated droplets and mist, as streaks, in the main pass. */
  readonly streaks: Mesh;
  /** Dense spray as sphere impostors, on the fluid layer (scene/fluidPass.ts). */
  readonly water: Mesh;
  private readonly renderer: WebGPURenderer;
  private readonly dt = uniform(FLUID.maxStep);
  private readonly frameDt = uniform(0);
  /** Frame number (wrapping), so each frame draws new random numbers. */
  private readonly frame = uniform(0);
  private readonly splashes = Array.from({ length: SPLASHES }, () => new Vector4(0, 0, 1, 0));
  private readonly splashMotion = Array.from({ length: SPLASHES }, () => new Vector4());
  private splashCount = 0;
  private readonly startNodes: ComputeNode[];
  private readonly stepNodes: ComputeNode[];
  private readonly landNode: ComputeNode;
  private readonly resetNode: ComputeNode;

  constructor(renderer: WebGPURenderer, coarse: boolean, sea: FluidSea) {
    this.renderer = renderer;
    const n = OCEAN_FLUID.particles(coarse);
    const grid = fluidGrid(OCEAN_FLUID.cell(coarse));
    const { res, dx, invDx } = grid;
    const cells = res * res * res;
    const { buffers, landings } = sea;

    // position and J; velocity and life left (0 when dead); then three per particle: the rows of C, with
    // kind (0 droplet, 1 mist), size and age in their w
    const pos = instancedArray(n, 'vec4') as Vec4Buffer;
    const vel = instancedArray(n, 'vec4') as Vec4Buffer;
    const aff = instancedArray(n * 3, 'vec4') as Vec4Buffer;
    const head = instancedArray(1, 'uint').toAtomic();
    // mass and momentum per cell, summed atomically in fixed point; then the cell's velocity and mass
    const gridSum = instancedArray(cells * 4, 'int').toAtomic() as StorageBufferNode<'int'>;
    const gridVel = instancedArray(cells, 'vec4') as Vec4Buffer;
    // how much water is around each particle (its own share counts about 0.3): dense spray is drawn as one
    // surface, isolated droplets as streaks
    const crowd = instancedArray(n, 'float') as StorageBufferNode<'float'>;
    // each vertex's direction and how high the land stands there, so spray comes down on the islands rather
    // than through them (one buffer: the landing pass is at WebGPU's eight)
    const dir = new Vector3();
    const terrain = instancedArray(
      Float32Array.from({ length: sea.field.count * 4 }, (_, k) => {
        const i = Math.floor(k / 4);
        if (k % 4 < 3) return sea.field.positions[i * 3 + (k % 4)]!;
        return sea.coastal ? Math.max(0, shoreGroundHeight(dir.fromArray(sea.field.positions, i * 3))) : 0;
      }),
      'vec4',
    ) as Vec4Buffer;
    const splashes = uniformArray(this.splashes, 'vec4') as UniformArrayNode<'vec4'>;
    const splashMotion = uniformArray(this.splashMotion, 'vec4') as UniformArrayNode<'vec4'>;

    const random = (seed: Node<'uint'>, k: number) => hash(seed.add(k));
    const radial = (x: Vec3) => normalize(x);
    /** The sea's surface radius at unit direction d: shape, swell, and the simulated height of a vertex. */
    const seaRadius = (d: Vec3, height: Float) => {
      const weather = oceanWeatherAt(d, sea.uniforms);
      const wave = oceanWaveVertex(d, sea.waves, sea.uniforms, weather, sea.field.spacing);
      return baseR(d).add(wave.height).add(height);
    };

    /** Starts one particle in the next slot of the ring; `seed` draws its randomness. */
    const start = (
      seed: Node<'uint'>,
      d: Vec3,
      origin: Float,
      travel: Vec3,
      strength: Float,
      vertex: Node<'uint'>,
    ) => {
      const slot = atomicAdd(head.element(0), uint(1)).mod(n);
      const mist = random(seed, 0).greaterThan(DROPLET_SHARE);
      const side = normalize(cross(d, select(d.y.abs().lessThan(0.9), vec3(0, 1, 0), vec3(1, 0, 0))));
      const side2 = cross(d, side);
      const angle = random(seed, 1).mul(Math.PI * 2);
      const ring = side.mul(angle.cos()).add(side2.mul(angle.sin()));
      const spread = random(seed, 2)
        .sqrt()
        .mul(select(mist, float(0.02), float(0.03)))
        .mul(strength);
      const p = d.mul(origin.add(random(seed, 3).mul(0.01))).add(ring.mul(spread));
      // thrown clear of the water around it (the old spray started 3.5 cm up; these start at the surface)
      const up = select(mist, float(0.32), float(0.48)).mul(strength).add(random(seed, 4).mul(0.2));
      const outward = random(seed, 5).mul(0.16).add(0.12).mul(strength);
      const v = d.mul(up).add(travel).add(ring.mul(outward));
      // mostly tiny: a droplet a few millimetres across is metres on a sea this size
      const size = select(
        mist,
        random(seed, 6).mul(0.0015).add(0.001),
        random(seed, 6).pow(4).mul(0.004).add(0.0012),
      );
      const life = select(mist, random(seed, 7).mul(0.55).add(0.28), float(6));
      pos.element(slot).assign(vec4(p, 1));
      vel.element(slot).assign(vec4(v, life));
      aff.element(slot.mul(3)).assign(vec4(0, 0, 0, select(mist, float(1), float(0))));
      aff.element(slot.mul(3).add(1)).assign(vec4(0, 0, 0, size));
      aff.element(slot.mul(3).add(2)).assign(vec4(0));
      // a droplet takes its water from the sea it leaves (mist is spent)
      If(mist.not(), () => {
        atomicAdd(
          landings.element(vertex.mul(2)),
          int(-Math.round(FLUID.splashVolume * FLUID.landing.volume)),
        );
      });
    };

    // Where the sea breaks, by the measures the foam flecks use: surf on a coast's breaking band, and water both
    // aerated and churning out at sea (dry vertices report neither).
    const emit = Fn(() => {
      const i = instanceIndex;
      const here = buffers.surface.element(i);
      const vertex = buffers.vertex.element(i);
      const aerated = here.y.greaterThanEqual(0.18).and(here.z.greaterThanEqual(0.1));
      const open = select(aerated, here.y.mul(0.82).add(here.z.mul(0.3)), float(0));
      const score = max(here.w, open);
      // surf pounding a bank throws far more spray than a whitecap
      const excess = max(
        clamp(open.sub(0.2).div(0.8), 0, 1),
        clamp(here.w.sub(0.1).div(0.4), 0, 1).mul(SURF_SPRAY),
      );
      const expected = excess
        .mul((EMISSION * n) / 32768)
        .mul(vertex.w)
        .mul(this.frameDt);
      const strength = clamp(score, 0.25, 0.95);
      const d = normalize(vertex.xyz);
      const flow = buffers.flow.element(i).xyz;
      const speed = length(flow);
      // carried on the way the water runs
      const travel = select(speed.greaterThan(1e-4), flow.div(speed), vec3(0)).mul(strength.mul(0.34));
      for (let k = 0; k < 2; k++) {
        const seed = uint(this.frame)
          .mul(sea.field.count)
          .add(i)
          .mul(16)
          .add(k * 8);
        If(random(seed, 7).lessThan(expected.sub(k)), () => {
          start(seed, d, seaRadius(d, here.x).add(0.008), travel, strength, i);
        });
      }
    })().compute(sea.field.count);

    // drag splashes: a burst at each, thrown the way the hand moved
    const thrown = Fn(() => {
      const e = instanceIndex.div(PER_SPLASH);
      const splash = splashes.element(e);
      If(splash.w.lessThanEqual(0), () => {
        Return();
      });
      const d = normalize(splash.xyz);
      const vertex = buffers.lookup.element(cubeCell(d, LOOKUP_SIZE));
      const seed = uint(this.frame)
        .mul(SPLASHES * PER_SPLASH)
        .add(instanceIndex)
        .mul(16)
        .add(7);
      const travel = splashMotion.element(e).xyz.mul(splash.w.mul(0.4));
      start(seed, d, seaRadius(d, buffers.surface.element(vertex).x).add(0.01), travel, splash.w, vertex);
    })().compute(SPLASHES * PER_SPLASH);

    // the three cells about a particle on each axis, and their B-spline weights (splineWeights)
    const stencil = (x: Vec3) => {
      const g = x.add(FLUID.extent).mul(invDx);
      const base = clamp(floor(g.sub(0.5)), 0, res - 3);
      const f = g.sub(base);
      const w0 = float(1.5).sub(f).pow(2).mul(0.5);
      const w1 = float(0.75).sub(f.sub(1).pow(2));
      const w2 = f.sub(0.5).pow(2).mul(0.5);
      const pick = (a: Node<'int'>, axis: 'x' | 'y' | 'z') =>
        select(a.equal(0), w0[axis], select(a.equal(1), w1[axis], w2[axis]));
      return { base, f, pick };
    };
    const neighbours = (x: Vec3, body: (cell: Node<'int'>, w: Float, dpos: Vec3) => void) => {
      const { base, f, pick } = stencil(x);
      Loop({ start: 0, end: 27, type: 'int' }, ({ i: o }) => {
        const a = o.mod(3);
        const b = o.div(3).mod(3);
        const c = o.div(9);
        const w = pick(a, 'x').mul(pick(b, 'y')).mul(pick(c, 'z'));
        const offset = vec3(float(a), float(b), float(c));
        const dpos = offset.sub(f).mul(dx);
        const cell = int(base.z.add(float(c)))
          .mul(res * res)
          .add(int(base.y.add(float(b))).mul(res))
          .add(int(base.x.add(float(a))));
        body(cell, w, dpos);
      });
    };
    const fixed = (x: Float) => int(round(x.mul(FLUID.fixed)));
    const live = (i: Node<'uint'>) => vel.element(i).w.greaterThan(0);
    const isMist = (i: Node<'uint'>) => aff.element(i.mul(3)).w.greaterThan(0.5);

    const clearGrid = Fn(() => {
      const c = instanceIndex.mul(4);
      for (let k = 0; k < 4; k++) atomicStore(gridSum.element(c.add(k)), int(0));
    })().compute(cells);

    // particles to grid: mass, momentum, and the pressure of each particle's squeezed or stretched volume
    const toGrid = Fn(() => {
      const i = instanceIndex;
      If(live(i).not().or(isMist(i)), () => {
        Return();
      });
      const x = pos.element(i).xyz;
      const v = vel.element(i).xyz;
      const J = pos.element(i).w;
      const stress = this.dt.mul(-4 * FLUID.stiffness * invDx * invDx).mul(J.sub(1));
      const rows = [0, 1, 2].map((r) => aff.element(i.mul(3).add(r)).xyz);
      neighbours(x, (cell, w, dpos) => {
        const momentum = vec3(dot(rows[0]!, dpos), dot(rows[1]!, dpos), dot(rows[2]!, dpos))
          .add(dpos.mul(stress))
          .add(v)
          .mul(w);
        atomicAdd(gridSum.element(cell.mul(4)), fixed(w));
        atomicAdd(gridSum.element(cell.mul(4).add(1)), fixed(momentum.x));
        atomicAdd(gridSum.element(cell.mul(4).add(2)), fixed(momentum.y));
        atomicAdd(gridSum.element(cell.mul(4).add(3)), fixed(momentum.z));
      });
    })().compute(n);

    // grid: velocity, radial gravity, and a still rim so nothing leaves the box
    const updateGrid = Fn(() => {
      const c = instanceIndex;
      const mass = float(atomicLoad(gridSum.element(c.mul(4)))).div(FLUID.fixed);
      const cell = gridVel.element(c);
      If(mass.lessThanEqual(0), () => {
        cell.assign(vec4(0));
        Return();
      });
      const momentum = vec3(
        float(atomicLoad(gridSum.element(c.mul(4).add(1)))),
        float(atomicLoad(gridSum.element(c.mul(4).add(2)))),
        float(atomicLoad(gridSum.element(c.mul(4).add(3)))),
      ).div(FLUID.fixed);
      const ijk = vec3(float(c.mod(res)), float(c.div(res).mod(res)), float(c.div(res * res)));
      const at = ijk.mul(dx).sub(FLUID.extent);
      const v = momentum
        .div(mass)
        .sub(radial(at).mul(this.dt.mul(FLUID.gravity)))
        .toVar();
      const rim = ijk.x
        .lessThan(2)
        .or(ijk.y.lessThan(2))
        .or(ijk.z.lessThan(2))
        .or(ijk.x.greaterThanEqual(res - 2))
        .or(ijk.y.greaterThanEqual(res - 2))
        .or(ijk.z.greaterThanEqual(res - 2));
      cell.assign(vec4(select(rim, vec3(0), v), mass));
    })().compute(cells);

    // grid to particles: velocity and its APIC gradient; mist just flies
    const toParticles = Fn(() => {
      const i = instanceIndex;
      If(live(i).not(), () => {
        Return();
      });
      const p = pos.element(i);
      const pv = vel.element(i);
      const x = p.xyz.toVar();
      If(isMist(i), () => {
        crowd.element(i).assign(0);
        const v = pv.xyz
          .sub(radial(x).mul(this.dt.mul(FLUID.gravity)))
          .mul(exp(this.dt.mul(-FLUID.mistDrag)));
        pv.assign(vec4(v, pv.w));
        p.assign(vec4(x.add(v.mul(this.dt)), p.w));
        Return();
      });
      const v = vec3(0).toVar();
      const rows = [0, 1, 2].map(() => vec3(0).toVar());
      const around = float(0).toVar();
      neighbours(x, (cell, w, dpos) => {
        const node = gridVel.element(cell);
        const g = node.xyz;
        around.addAssign(node.w.mul(w));
        v.addAssign(g.mul(w));
        (['x', 'y', 'z'] as const).forEach((axis, r) =>
          rows[r]!.addAssign(dpos.mul(g[axis].mul(w).mul(4 * invDx * invDx))),
        );
      });
      crowd.element(i).assign(around);
      const trace = rows[0]!.x.add(rows[1]!.y).add(rows[2]!.z);
      const J = clamp(p.w.mul(trace.mul(this.dt).add(1)), FLUID.minJ, FLUID.maxJ);
      const moved = v.mul(exp(this.dt.mul(-FLUID.drag)));
      const next = x.add(moved.mul(this.dt));
      // anything about to leave the grid is gone
      const inside = max(max(next.x.abs(), next.y.abs()), next.z.abs()).lessThan(FLUID.extent - 3 * dx);
      p.assign(vec4(next, J));
      pv.assign(vec4(moved, select(inside, pv.w, float(0))));
      for (let r = 0; r < 3; r++) {
        const row = aff.element(i.mul(3).add(r));
        row.assign(vec4(rows[r]!, row.w));
      }
    })().compute(n);

    // Once a frame: particles age, and those that have come down onto the water or the land end there. A
    // droplet gives its water back to the vertex beneath, with some foam; a droplet that runs out of life in the
    // air gives its water back too, so the sea keeps every drop.
    const land = Fn(() => {
      const i = instanceIndex;
      const pv = vel.element(i);
      If(pv.w.lessThanEqual(0), () => {
        Return();
      });
      const x = pos.element(i).xyz;
      const d = normalize(x);
      const vertex = buffers.lookup.element(cubeCell(d, LOOKUP_SIZE));
      const water = seaRadius(d, buffers.surface.element(vertex).x);
      // the land's height blended over the vertex and its neighbours, as sampleSurface blends the water: the
      // terrain is steep, and one vertex's height would leave spray standing on air or inside the rock
      const terrainHeight = float(0).toVar();
      const weights = float(0).toVar();
      Loop({ start: 0, end: 7, type: 'uint' }, ({ i: k }) => {
        const j = select(k.equal(0), vertex, buffers.neighbor.element(vertex.mul(6).add(k.sub(1))));
        const ground = terrain.element(j);
        const offset = ground.xyz.sub(d);
        const w = float(1).div(dot(offset, offset).add(0.0002));
        terrainHeight.addAssign(ground.w.mul(w));
        weights.addAssign(w);
      });
      const ground = baseR(d).add(terrainHeight.div(weights));
      const age = aff.element(i.mul(3).add(2));
      age.w.addAssign(this.frameDt);
      // under the surface while coming down; or, once clear of its launch, resting on the water (droplets that
      // land together hold each other up)
      const settled = age.w.greaterThan(0.15);
      const falling = dot(pv.xyz, d).lessThan(0).or(settled);
      const down = length(x)
        .lessThan(max(water, ground).add(select(settled, float(0.006), float(0))))
        .and(falling);
      const expired = pv.w.sub(this.frameDt).lessThanEqual(0);
      If(down.or(expired), () => {
        const mist = isMist(i);
        If(mist.not(), () => {
          atomicAdd(
            landings.element(vertex.mul(2)),
            int(Math.round(FLUID.splashVolume * FLUID.landing.volume)),
          );
        });
        const foam = select(mist, float(LANDING_FOAM.mist), float(LANDING_FOAM.droplet)).mul(
          select(down, float(1), float(0)),
        );
        atomicAdd(landings.element(vertex.mul(2).add(1)), int(round(foam.mul(FLUID.landing.foam))));
        pv.w.assign(0);
        Return();
      });
      pv.w.assign(pv.w.sub(this.frameDt));
    })().compute(n);

    this.startNodes = [emit, thrown];
    this.stepNodes = [clearGrid, toGrid, updateGrid, toParticles];
    this.landNode = land;
    this.resetNode = Fn(() => {
      vel.element(instanceIndex).assign(vec4(0));
    })().compute(n);

    // Drawn straight from storage, as camera-facing quads. Spray dense enough to read as water goes to the fluid
    // layer as sphere impostors, which scene/fluidPass.ts smooths into one surface; isolated droplets and mist are
    // streaks along their motion (the blur any eye or camera sees in fast spray), clear and bright at the head.
    const posView = storage(pos.value, 'vec4', n).toReadOnly() as Vec4Buffer;
    const velView = storage(vel.value, 'vec4', n).toReadOnly() as Vec4Buffer;
    const affView = storage(aff.value, 'vec4', n * 3).toReadOnly() as Vec4Buffer;
    const crowdView = storage(crowd.value, 'float', n).toReadOnly() as StorageBufferNode<'float'>;
    const quad = () => {
      const plane = new PlaneGeometry(2, 2);
      const geometry = new InstancedBufferGeometry();
      geometry.setIndex(plane.getIndex());
      geometry.setAttribute('position', plane.getAttribute('position'));
      geometry.instanceCount = n;
      return geometry;
    };
    const corner = varyingProperty('vec2', 'vCorner');
    const weight = varyingProperty('float', 'vWeight');
    const centre = varyingProperty('vec3', 'vCentre');
    const radius = varyingProperty('float', 'vRadius');
    /** 0 for an isolated droplet, 1 for one in dense spray. */
    const dense = (i: Node<'uint'>) => smoothstep(DENSE * 0.6, DENSE * 1.2, crowdView.element(i));
    const alive = (i: Node<'uint'>) => velView.element(i).w.greaterThan(0);
    const mistAt = (i: Node<'uint'>) => affView.element(i.mul(3)).w.greaterThan(0.5);

    const streak = new MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: DoubleSide });
    streak.vertexNode = Fn(() => {
      const i = instanceIndex;
      const life = velView.element(i).w;
      const mist = mistAt(i);
      const size = affView.element(i.mul(3).add(1)).w;
      const at = modelViewMatrix.mul(vec4(posView.element(i).xyz, 1));
      const motion = modelViewMatrix.mul(vec4(velView.element(i).xyz, 0)).xy;
      const speed = length(motion);
      const along = select(speed.greaterThan(1e-5), motion.div(speed), vec2(1, 0));
      const across = vec2(along.y.negate(), along.x);
      const reach = size.add(speed.mul(STREAK_TIME));
      const fade = clamp(life.div(select(mist, float(0.22), float(0.15))), 0, 1);
      const shown = select(alive(i), float(1), float(0)).mul(select(mist, float(1), float(1).sub(dense(i))));
      weight.assign(fade.mul(shown).mul(select(mist, float(MIST_OPACITY), float(DROPLET_OPACITY))));
      corner.assign(positionGeometry.xy);
      const offset = along
        .mul(positionGeometry.x.mul(reach))
        .add(across.mul(positionGeometry.y.mul(size)))
        .mul(select(shown.greaterThan(0), float(1), float(0)));
      return cameraProjectionMatrix.mul(at.add(vec4(offset, 0, 0)));
    })();
    streak.colorNode = Fn(() => {
      // soft across, fading from a bright head to a faint tail
      const side = float(1).sub(corner.y.mul(corner.y));
      const head = corner.x.add(1).mul(0.5);
      const alpha = side
        .mul(mix(float(0.25), float(1), head))
        .mul(float(1).sub(smoothstep(0.7, 1, corner.x.abs())));
      return vec4(0.86, 0.95, 1, alpha.mul(weight));
    })();
    this.streaks = new Mesh(quad(), streak);
    this.streaks.frustumCulled = false;

    const impostor = new MeshBasicNodeMaterial();
    impostor.vertexNode = Fn(() => {
      const i = instanceIndex;
      const take = alive(i).and(mistAt(i).not());
      const r = select(take, dense(i).mul(IMPOSTOR_RADIUS), float(0));
      const at = modelViewMatrix.mul(vec4(posView.element(i).xyz, 1));
      centre.assign(at.xyz);
      radius.assign(r);
      corner.assign(positionGeometry.xy);
      return cameraProjectionMatrix.mul(at.add(vec4(positionGeometry.xy.mul(r), 0, 0)));
    })();
    // the sphere's surface toward the camera: its view distance, and its depth so the nearest wins
    const surfacePoint = Fn(() => {
      const r2 = dot(corner, corner);
      Discard(r2.greaterThan(1).or(radius.lessThanEqual(0)));
      return centre.add(vec3(corner.mul(radius), float(1).sub(r2).sqrt().mul(radius)));
    });
    impostor.colorNode = Fn(() => vec4(surfacePoint().z.negate(), 1, 0, 1))();
    impostor.depthNode = Fn(() => {
      const clip = cameraProjectionMatrix.mul(vec4(surfacePoint(), 1));
      return clip.z.div(clip.w);
    })();
    this.water = new Mesh(quad(), impostor);
    this.water.frustumCulled = false;
    this.water.layers.set(FLUID_LAYER);
  }

  /** A drag throwing water at dir, the way the hand is moving. */
  splash(dir: Vector3, strength: number, motion: Vector3): void {
    const slot = this.splashCount++ % SPLASHES;
    this.splashes[slot]!.set(dir.x, dir.y, dir.z, strength);
    this.splashMotion[slot]!.set(motion.x, motion.y, motion.z, 0);
  }

  clear(): void {
    this.renderer.compute(this.resetNode);
  }

  update(dt: number): void {
    if (dt <= 0) return;
    const steps = Math.min(6, Math.max(1, Math.ceil(dt / FLUID.maxStep)));
    this.dt.value = dt / steps;
    this.frameDt.value = dt;
    this.frame.value = (this.frame.value + 1) % 4096;
    const passes = [...this.startNodes];
    for (let s = 0; s < steps; s++) passes.push(...this.stepNodes);
    passes.push(this.landNode);
    this.renderer.compute(passes);
    for (const splash of this.splashes) splash.w = 0;
  }
}
