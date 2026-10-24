import { BufferAttribute, Vector2, Vector3, Vector4, type BufferGeometry } from 'three';
import type { ComputeNode, Node, StorageBufferNode, UniformArrayNode, WebGPURenderer } from 'three/webgpu';
import {
  abs,
  atomicAdd,
  atomicLoad,
  atomicStore,
  clamp,
  cross,
  dot,
  exp,
  float,
  Fn,
  If,
  instancedArray,
  instanceIndex,
  int,
  length,
  Loop,
  max,
  min,
  mix,
  Return,
  round,
  select,
  sign,
  sin,
  smoothstep,
  storage,
  uint,
  uniform,
  uniformArray,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { createOceanField, type OceanField } from '../core/oceanField';
import { FLUID } from '../core/oceanFluid';
import { OCEAN_DEFAULTS, type OceanTuning } from '../core/oceanTuning';
import { OCEAN_DEPTH, OCEAN_WIND_ANGLE } from '../core/oceanFFT';
import { WIND, windAt as windOnOrb } from '../core/oceanWind';
import { SHELF_EDGE, SHORE_SITES, shoreSwell, SURF_REACH, SURF_SET_WAVES } from '../core/oceanShores';
import { SURFACE } from '../core/shallowWater';
import { WAVE_SLOTS } from '../core/waveKernel';
import { oceanWeatherAt, type DeformUniforms } from '../nodes/deform';
import { driftBasis, windAt } from '../nodes/oceanSurface';
import { oceanWaveFoam, type OceanWaveFields } from '../nodes/oceanWaves';

/** A push on the water: a tap or release (no motion), or a drag step (with motion along the surface). */
export interface OceanForce {
  dir: Vector3;
  strength: number;
  motion?: Vector3;
}

/** A finger gradually pushes water down; release gives the loaded surface a small upward kick. */
export const oceanPressLoad = (seconds: number): number => 10 * (1 - Math.exp(-Math.max(0, seconds) / 0.5));
export const oceanReleaseImpulse = (seconds: number): number =>
  1.6 * (1 - Math.exp(-Math.max(0, seconds) / 0.7));

/** How the hand of god moves water. */
const HAND = {
  /** Vertical speed a tap of strength 1 gives the water under it, orb radii per second. */
  impulse: 1.4,
  /**
   * Sharpness k of a tap's footprint (see hat): pressing down out to about 0.08 radians. A narrow, hard push
   * throws a tall, short ring; a broad one only a long, low swell that hides in the sea.
   */
  tapSharpness: 300,
  /** A drag churns a narrower strip. */
  dragSharpness: 110,
  /** Downward acceleration per unit of press load. */
  press: 0.008,
  pressSharpness: 40,
} as const;

/**
 * A push's footprint at x = (1 - cos angle) * sharpness: down in the middle, up in a rim around it, and zero in
 * total (it's the Laplacian of a Gaussian), the way a hand moves water aside rather than taking any away.
 * Without the rim, a push's broadest part excites the orb's slowest swaying, which sinks for many seconds.
 */
const hat = (x: number): number => (1 - x) * Math.exp(-x);
const hatNode = (x: Float): Float => float(1).sub(x).mul(exp(x.negate()));

const STEP = 1 / 60;
const MAX_STEPS = 4;
const EVENTS = 4;
/** Fixed-point scale for summing the sea's mean height: 1e-10 resolution, room for means up to 0.2. */
const VOLUME_FIXED = 1e10;
/**
 * Seconds each foam drift runs before it starts over. Foam's pattern is read at its vertex moved back by the
 * drift; a drift that ran forever would shear it into fingerprints wherever the current swirls, so two drifts,
 * half a cycle apart, each restart while hidden and the shader crossfades them (flow maps, Vlachos 2010).
 */
const DRIFT_PERIOD = 1;
/**
 * Farthest a foam drift goes, orb radii: a couple of the pattern's own grains. The foam itself rides the current
 * in the sim; the pattern only has to look carried, and a swirl's shear can't wind it into rings.
 */
const DRIFT_REACH = 0.025;

/** How the FFT's whitecaps (their remembered folding, averaged over a vertex) leave foam in the sim. */
const WHITECAP = {
  from: 0.25,
  to: 0.7,
  /** The foam a vertex keeps while its water is breaking. */
  foam: 0.45,
} as const;

/** The FFT waves, for whitecap foam; the ocean material's uniforms carry their tuning and weather. */
export interface SimWaves {
  fields: OceanWaveFields;
  uniforms: DeformUniforms;
}

type Vec4Buffer = StorageBufferNode<'vec4'>;
type Float = Node<'float'>;
type Vec3 = Node<'vec3'>;

/**
 * The interactive sea on the GPU: the step in core/shallowWater.ts (iWave with the finite-depth operator,
 * blended by depth into the nonlinear shallow-water equations), plus foam and turbulence that ride the
 * current, and scripted surf on coasts. Splashes (materials/oceanFluid.ts) take water and give it back through
 * `landings`. The mesh and coast are fixed at creation; rendering reads the results from storage (renderBuffers).
 */
export class OceanSim {
  readonly count: number;
  readonly spacing: number;
  readonly field: OceanField;

  private readonly renderer: WebGPURenderer;
  private readonly uniforms = {
    dt: uniform(STEP),
    frameDt: uniform(STEP),
    time: uniform(0),
    wind: uniform(OCEAN_DEFAULTS.windSpeed),
    /** Where the wind blows, as OCEAN_WIND_ANGLE (core/oceanWind.ts). */
    windAngle: uniform(OCEAN_WIND_ANGLE),
    amplitude: uniform(OCEAN_DEFAULTS.amplitude),
    /** The surf's clock: the swell's frequency times time, added up frame by frame, never time times a speed. */
    surfClock: uniform(0),
    /** The sets' clock: the swell's frequency over SURF_SET_WAVES, added up the same way. */
    setClock: uniform(0),
    /** The swell's strength on the shores, following the wind slowly. */
    surfEnergy: uniform(OCEAN_DEFAULTS.windSpeed * OCEAN_DEFAULTS.amplitude),
    /** xyz direction, w how hard (press load). */
    press: uniform(new Vector4(0, 0, 1, 0)),
    /** The pushes' mean over the sea, taken back everywhere so the hand moves water rather than removing it. */
    pushMean: uniform(0),
    /** 0 where a foam drift restarts this frame, 1 where it keeps running. */
    driftKeep: uniform(new Vector2(1, 1)),
  };
  /** How much of the foam pattern follows the first drift (the second takes the rest). */
  private readonly driftBlend = uniform(0.5);
  /** Where the foam drifts are in their cycle, 0 to 1. */
  private driftClock = 0;
  /** The swell each island's surf comes from: its heading at the island (as a unit tangent there). */
  private readonly swellHeadings = SHORE_SITES.map(() => new Vector4());
  /** The swell's frequency and open-water wavenumber, its direction and strength (each easing after the wind's). */
  private readonly swell: { omega: number; k: number };
  private swellAngle = OCEAN_WIND_ANGLE;
  private swellSpeed: number = OCEAN_DEFAULTS.windSpeed;
  private readonly events = Array.from({ length: EVENTS }, () => new Vector4(0, 0, 1, 0));
  private readonly eventShapes = Array.from({ length: EVENTS }, () => new Vector4());

  /**
   * Splash water landing on (or leaving) each vertex since the last update, two per vertex in fixed point
   * (FLUID.landing in core/oceanFluid.ts): volume, and foam. The fluid adds to it atomically; the sim spends it.
   */
  readonly landings: StorageBufferNode<'int'>;
  private readonly surface: Vec4Buffer;
  private readonly slope: Vec4Buffer;
  private readonly drift: Vec4Buffer;
  /** The current after the last frame, in one place (the stepping copies alternate). */
  private readonly flow: Vec4Buffer;
  private readonly ground: Vec4Buffer;
  private readonly current: [Vec4Buffer, Vec4Buffer];
  private readonly lookup: StorageBufferNode<'uint'>;
  private readonly vertex: Vec4Buffer;
  private readonly neighbor: StorageBufferNode<'uint'>;
  private readonly stepNodes: [ComputeNode[], ComputeNode[]];
  private readonly finishNodes: [ComputeNode[], ComputeNode[]];
  private readonly resetNodes: ComputeNode[];
  private parity = 0;
  private readonly wetArea: number;

  constructor(
    renderer: WebGPURenderer,
    geometry: BufferGeometry,
    coastal = false,
    tuning: Readonly<OceanTuning> = OCEAN_DEFAULTS,
    waves?: SimWaves,
  ) {
    this.renderer = renderer;
    const field = createOceanField(geometry, coastal);
    this.field = field;
    this.count = field.count;
    this.spacing = field.spacing;
    this.wetArea = field.areas.reduce((sum, area, i) => sum + (field.wet[i] ? area : 0), 0);
    this.uniforms.wind.value = tuning.windSpeed;
    this.uniforms.amplitude.value = tuning.amplitude;
    this.swell = shoreSwell(tuning.peakWavelength);
    this.swellSpeed = tuning.windSpeed;
    if (coastal) {
      // WebGPU guarantees eight vertex buffers; the shore normal and distance share one attribute
      const shoreData = new Float32Array(this.count * 4);
      for (let i = 0; i < this.count; i++) {
        shoreData.set(
          [
            field.shoreNormal[i * 3]!,
            field.shoreNormal[i * 3 + 1]!,
            field.shoreNormal[i * 3 + 2]!,
            field.shoreDistance[i]!,
          ],
          i * 4,
        );
      }
      geometry.setAttribute('oceanShoreData', new BufferAttribute(shoreData, 4));
    }

    const n = this.count;
    const pack = (fill: (i: number) => [number, number, number, number]) => {
      const out = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) out.set(fill(i), i * 4);
      return out;
    };
    const p = field.positions;
    this.vertex = instancedArray(
      pack((i) => [p[i * 3]!, p[i * 3 + 1]!, p[i * 3 + 2]!, field.areas[i]!]),
      'vec4',
    ) as Vec4Buffer;
    this.neighbor = instancedArray(field.neighbors, 'uint') as StorageBufferNode<'uint'>;
    // Each operator slot as (vertex index, weight), indices exact as floats. WebGPU guarantees only eight
    // storage buffers per shader stage, so the inputs are packed tightly.
    const slots = new Float32Array(field.kernel.indices.length * 2);
    for (let s = 0; s < field.kernel.indices.length; s++) {
      slots[s * 2] = field.kernel.indices[s]!;
      slots[s * 2 + 1] = field.kernel.weights[s]!;
    }
    const kernel = instancedArray(slots, 'vec2') as StorageBufferNode<'vec2'>;
    // depth (0 on land), shore distance, the shelf's lag (core/oceanShores.ts), foam grain
    const ground = instancedArray(
      pack((i) => [
        field.wet[i] ? field.depth[i]! : 0,
        field.shoreDistance[i]!,
        field.shelfLag[i]!,
        field.foamGrain[i]!,
      ]),
      'vec4',
    ) as Vec4Buffer;
    const shore = instancedArray(
      pack((i) => [
        field.shoreNormal[i * 3]!,
        field.shoreNormal[i * 3 + 1]!,
        field.shoreNormal[i * 3 + 2]!,
        field.shoreSite[i]!,
      ]),
      'vec4',
    ) as Vec4Buffer;
    // how strongly each spot answers a push
    const forceGrains = instancedArray(field.forceGrain, 'float') as StorageBufferNode<'float'>;
    this.lookup = instancedArray(field.lookup, 'uint') as StorageBufferNode<'uint'>;
    // h, h a step ago, foam, turbulence; and the tangential current. Both ping-pong between steps.
    const heights: [Vec4Buffer, Vec4Buffer] = [
      instancedArray(n, 'vec4') as Vec4Buffer,
      instancedArray(n, 'vec4') as Vec4Buffer,
    ];
    this.current = [instancedArray(n, 'vec4') as Vec4Buffer, instancedArray(n, 'vec4') as Vec4Buffer];
    // what rendering reads: height with surf, foam, turbulence, surf breaking; slope; foam drift
    this.surface = instancedArray(n, 'vec4') as Vec4Buffer;
    this.slope = instancedArray(n, 'vec4') as Vec4Buffer;
    this.drift = instancedArray(n, 'vec4') as Vec4Buffer;
    this.flow = instancedArray(n, 'vec4') as Vec4Buffer;
    this.ground = ground;
    this.landings = instancedArray(n * 2, 'int').toAtomic() as StorageBufferNode<'int'>;

    const u = this.uniforms;
    const events = uniformArray(this.events, 'vec4') as UniformArrayNode<'vec4'>;
    const eventShapes = uniformArray(this.eventShapes, 'vec4') as UniformArrayNode<'vec4'>;
    const wetAt = (j: Node<'uint'>) => ground.element(j).x.greaterThan(0);
    /**
     * The wind's slow surface drift at vertex j, dying away against the bank. The stored current is the waves'
     * alone, so a wind that turns or drops moves no water; this is added wherever the whole flow is wanted.
     */
    const windDrift = (j: Node<'uint'>): Vec3 =>
      windAt(this.vertex.element(j).xyz, u.windAngle)
        .mul(u.wind.mul(WIND.drift))
        .mul(smoothstep(0, 0.06, ground.element(j).y));
    const invSqrt3 = 1 / Math.sqrt(3);
    const meanArea = field.areas.reduce((sum, area) => sum + area, 0) / n;
    const invSpacing = 1 / field.spacing;

    /** Vertical acceleration from taps, releases, drags (each spread over the frame) and a held press. */
    const pushAt = (q: Vec3, forceGrain: Float): Float => {
      let accel: Float = float(0);
      for (let e = 0; e < EVENTS; e++) {
        const event = events.element(e);
        const sharpness = eventShapes.element(e).w;
        const falloff = hatNode(float(1).sub(dot(q, event.xyz)).mul(sharpness));
        accel = accel.add(event.w.mul(HAND.impulse).mul(falloff).mul(forceGrain).div(u.frameDt));
      }
      const press = hatNode(float(1).sub(dot(q, u.press.xyz)).mul(HAND.pressSharpness))
        .mul(u.press.w)
        .mul(HAND.press);
      return accel.sub(press).sub(u.pushMean);
    };

    /** Drags stir the current: outward from the finger and along its motion; taps swirl it a little. */
    const stirAt = (q: Vec3): Vec3 => {
      let stir: Vec3 = vec3(0);
      for (let e = 0; e < EVENTS; e++) {
        const event = events.element(e);
        const shape = eventShapes.element(e);
        const alignment = dot(q, event.xyz);
        const gain = abs(event.w).mul(exp(alignment.sub(1).mul(shape.w)));
        const away = q.sub(event.xyz.mul(alignment));
        const along = select(length(shape.xyz).greaterThan(1e-4), shape.xyz, cross(event.xyz, q));
        stir = stir.add(away.div(length(away).add(1e-5)).mul(0.055).add(along.mul(0.025)).mul(gain));
      }
      return stir;
    };

    const momentum = (from: number) =>
      Fn(() => {
        const i = instanceIndex;
        const out = this.current[1 - from]!.element(i);
        const g4 = ground.element(i);
        If(g4.x.lessThanEqual(0), () => {
          out.assign(vec4(0));
          Return();
        });
        const v = this.vertex.element(i);
        const q = v.xyz;
        const hi = heights[from]!.element(i).x;
        // grad h from edge differences; a dry neighbour adds no slope
        const gradient = vec3(0).toVar();
        Loop({ start: 0, end: 6, type: 'uint' }, ({ i: k }) => {
          const j = this.neighbor.element(i.mul(6).add(k));
          If(j.notEqual(i).and(wetAt(j)), () => {
            const chord = this.vertex.element(j).xyz.sub(q);
            gradient.addAssign(
              chord.mul(
                heights[from]!.element(j)
                  .x.sub(hi)
                  .mul(0.5 * invSqrt3),
              ),
            );
          });
        });
        // The waves' current (the wind's drift is added on top where needed): it carries water (heightStep), is
        // pushed by gravity and the hand, turned back at the bank and capped.
        const drag = exp(u.dt.mul(-SURFACE.drag));
        const waves = this.current[from]!.element(i)
          .xyz.mul(drag)
          .sub(gradient.mul(u.dt.mul(SURFACE.gravity)).div(v.w))
          .add(stirAt(q).mul(u.dt.div(u.frameDt)))
          .toVar();
        if (coastal) {
          // currents running into the bank turn back
          const normal = shore.element(i).xyz;
          const incoming = min(dot(waves, normal), 0);
          waves.subAssign(normal.mul(incoming.mul(float(1).sub(smoothstep(0, 0.06, g4.y))).mul(1.15)));
        }
        waves.subAssign(q.mul(dot(waves, q)));
        const speed = length(waves);
        out.assign(vec4(waves.mul(min(speed, SURFACE.maxCurrent).div(max(speed, 1e-6))), 0));
      })().compute(n);

    const heightStep = (from: number) =>
      Fn(() => {
        const i = instanceIndex;
        const out = heights[1 - from]!.element(i);
        const g4 = ground.element(i);
        If(g4.x.lessThanEqual(0), () => {
          out.assign(vec4(0));
          Return();
        });
        const state = heights[from]!.element(i).toVar();
        const h = state.x;
        const v = this.vertex.element(i);
        const q = v.xyz;
        const forceGrain = forceGrains.element(i);
        const foamGrain = g4.w;

        // iWave's operator, with dry neighbours mirroring this vertex so the coast reflects
        const operator = float(0).toVar();
        Loop({ start: 0, end: WAVE_SLOTS, type: 'uint' }, ({ i: s }) => {
          const slot = kernel.element(i.mul(WAVE_SLOTS).add(s));
          const j = uint(slot.x);
          operator.addAssign(slot.y.mul(select(wetAt(j), heights[from]!.element(j).x, h)));
        });

        // the waves' current carries water; foam rides it and the wind's drift too
        const waves = this.current[1 - from]!.element(i).xyz;
        const flow = waves.add(windDrift(i));
        const qi = g4.x.add(h);
        const rise = float(0).toVar();
        const spreadH = float(0).toVar();
        const wetNeighbours = float(0).toVar();
        const outflow = float(0).toVar();
        const sumFoam = float(0).toVar();
        const sumDelta = float(0).toVar();
        const upwindFoam = float(0).toVar();
        const upwindWeight = float(0).toVar();
        const spread = float(0).toVar();
        Loop({ start: 0, end: 6, type: 'uint' }, ({ i: k }) => {
          const j = this.neighbor.element(i.mul(6).add(k));
          const other = heights[from]!.element(j);
          const wet = wetAt(j);
          sumFoam.addAssign(select(wet, other.z, state.z));
          If(j.notEqual(i).and(wet), () => {
            const chord = this.vertex.element(j).xyz.sub(q);
            const otherWaves = this.current[1 - from]!.element(j).xyz;
            rise.addAssign(other.x.sub(other.y));
            spreadH.addAssign(other.x.sub(h));
            wetNeighbours.addAssign(1);
            // shallow-water flux through the shared face: the edge's mean current (the waves' part only; the
            // wind's drift moves foam, not the water column) times the column on the side it flows from
            const velocity = dot(waves.add(otherWaves), chord).mul(0.5);
            const column = max(select(velocity.greaterThan(0), qi, ground.element(j).x.add(other.x)), 0);
            outflow.addAssign(velocity.mul(column).mul(invSqrt3));
            sumDelta.addAssign(abs(other.x.sub(h)));
            // foam arrives from neighbours upstream of the current
            const weight = max(dot(chord, flow).negate(), 0);
            upwindFoam.addAssign(other.z.mul(weight));
            upwindWeight.addAssign(weight);
            spread.addAssign(
              dot(otherWaves.sub(waves), chord)
                .div(max(dot(chord, chord), 1e-5))
                .div(6),
            );
          });
        });

        const a = u.dt.mul(SURFACE.damping * 0.5);
        const ownRise = h.sub(state.y);
        const eased = ownRise.add(
          select(
            wetNeighbours.greaterThan(0),
            rise.div(max(wetNeighbours, 1)).sub(ownRise).mul(SURFACE.viscosity),
            float(0),
          ),
        );
        const force = pushAt(q, forceGrain).mul(u.dt).mul(u.dt);
        // a hand's reach, shorter over the shallows (pushRoom in core/shallowWater.ts)
        const reach = g4.x.mul(SURFACE.handReach / OCEAN_DEPTH);
        const push = force.mul(clamp(float(1).sub(h.mul(sign(force)).div(reach)), 0, 1));
        const dt2g = u.dt.mul(u.dt).mul(SURFACE.gravity);
        const wave = h
          .mul(2)
          .sub(h.sub(eased).mul(float(1).sub(a)))
          .sub(operator.mul(dt2g))
          .add(push)
          .div(a.add(1));
        // conservative smoothing between neighbours (SURFACE.shallowSmoothing)
        const smoothing = float(SURFACE.shallowSmoothing * meanArea).div(v.w);
        const shallow = h.sub(u.dt.mul(outflow).div(v.w)).add(spreadH.mul(smoothing)).add(push);
        const mixShallow = float(1).sub(smoothstep(0.008, 0.025, g4.x));
        // never below the bed in the shallows (surfaceFloor in core/shallowWater.ts)
        const floor = mix(float(-SURFACE.maxHeight), g4.x.negate(), mixShallow);
        const next = clamp(mix(wave, shallow, mixShallow), floor, SURFACE.maxHeight).toVar();

        // whitecaps on steep, raised crests; foam rides the current, spreads a little and fades
        const steepness = sumDelta.div(6).mul(invSpacing);
        const breaking = smoothstep(0.09, 0.22, steepness)
          .mul(smoothstep(0.001, 0.015, next))
          .mul(smoothstep(0.3, 0.8, foamGrain));
        const travel = clamp(u.dt.mul(length(flow)).mul(invSpacing * 0.65), 0, 0.35);
        const carried = state.z
          .mul(float(1).sub(travel))
          .add(upwindFoam.div(max(upwindWeight, 1e-5)).mul(travel));
        const fade = coastal
          ? select(g4.y.lessThan(0.15), exp(u.dt.mul(-1.7)), exp(u.dt.mul(-0.8)))
          : exp(u.dt.mul(-0.8));
        const foam = clamp(
          carried
            .mul(fade)
            .add(sumFoam.div(6).sub(state.z).mul(u.dt).mul(0.25))
            .add(breaking.mul(u.dt).mul(3.5)),
          0,
          1,
        );
        const lift = abs(next.sub(h).div(u.dt));
        const turbulence = clamp(
          max(
            state.w.mul(exp(u.dt.mul(-0.9))),
            steepness.mul(0.9).add(lift.mul(0.15)).add(abs(spread).mul(0.02)),
          ),
          0,
          1,
        );
        out.assign(vec4(next, h, foam, turbulence));
      })().compute(n);

    /**
     * Surf: the swell running in from its own direction, slowing and crowding on the shelf and swinging round to
     * break along the bank (shoreSurfPhase and sampleShoreSurf in core/oceanShores.ts).
     */
    const headings = uniformArray(this.swellHeadings, 'vec4') as UniformArrayNode<'vec4'>;
    const centres = uniformArray(
      SHORE_SITES.map((site) => new Vector4(site.center.x, site.center.y, site.center.z, 0)),
      'vec4',
    ) as UniformArrayNode<'vec4'>;
    const surfAt = (distance: Float, lag: Float, site: Float, q: Vec3, normal: Vec3) => {
      const smooth = (x: Float) => smoothstep(0, 1, x);
      const { omega, k } = this.swell;
      const heading = headings.element(int(site)).xyz;
      const facing = dot(heading, normal.negate());
      const across = max(float(SHELF_EDGE).sub(distance), 0);
      const spatial = dot(q.sub(centres.element(int(site)).xyz), heading)
        .mul(k)
        .add(lag.mul(omega))
        .sub(across.mul(k).mul(max(facing, 0.15)));
      const phase = spatial.sub(u.surfClock);
      // sets of bigger waves riding in with the crests, at each island's own times (surfSet)
      const lull = sin(spatial.div(SURF_SET_WAVES).sub(u.setClock).add(site.mul(2.4)))
        .mul(0.5)
        .add(0.5);
      const set = lull.mul(lull).mul(0.9).add(0.55);
      // shelfDepth (core/oceanShores.ts)
      const depth = clamp(smooth(distance.div(0.045)).mul(0.085).sub(0.004), 0.002, OCEAN_DEPTH);
      const envelope = float(1)
        .sub(smooth(distance.sub(0.04).div(SURF_REACH - 0.04)))
        .mul(min(float(OCEAN_DEPTH).div(depth).pow(0.25), 1.8))
        .mul(smooth(distance.div(0.025)).mul(0.35).add(0.65));
      const exposure = smooth(facing.add(0.2).div(0.8)).mul(0.85).add(0.15);
      const s = sin(phase);
      const crest = max(s, 0).pow(5);
      const strength = exposure.mul(u.surfEnergy).mul(set);
      const height = s.mul(0.32).add(crest.mul(0.95)).sub(0.17).mul(0.03).mul(envelope).mul(strength);
      const breaking = crest.mul(float(1).sub(smooth(distance.sub(0.012).div(0.06)))).mul(strength);
      const inside = distance.greaterThanEqual(0).and(distance.lessThanEqual(SURF_REACH));
      return { height: select(inside, height, float(0)), breaking: select(inside, breaking, float(0)) };
    };

    const finishSurface = (final: number) =>
      Fn(() => {
        const i = instanceIndex;
        const g4 = ground.element(i);
        If(g4.x.lessThanEqual(0), () => {
          this.surface.element(i).assign(vec4(0));
          Return();
        });
        const q = this.vertex.element(i).xyz;
        const state = heights[final]!.element(i);
        const foam = state.z.toVar();
        const turbulence = state.w.toVar();
        const surfHeight = float(0).toVar();
        const impact = float(0).toVar();
        if (waves) {
          // whitecaps leave foam on the water, which then rides the current (a tap's ring sweeps it aside)
          const memory = oceanWaveFoam(
            q,
            waves.fields,
            waves.uniforms,
            oceanWeatherAt(q, waves.uniforms),
            field.spacing,
          );
          const offshore = coastal ? smoothstep(0, 0.1, g4.y) : float(1);
          const whitecap = smoothstep(WHITECAP.from, WHITECAP.to, memory).mul(offshore);
          foam.assign(max(foam, whitecap.mul(WHITECAP.foam)));
        }
        if (coastal) {
          const surf = surfAt(g4.y, g4.z, shore.element(i).w, q, shore.element(i).xyz);
          surfHeight.assign(surf.height);
          impact.assign(
            clamp(
              surf.breaking.add(
                smoothstep(0.005, 0.045, state.x).mul(float(1).sub(smoothstep(0.02, 0.08, g4.y))),
              ),
              0,
              1,
            ),
          );
          foam.assign(clamp(foam.add(impact.mul(u.frameDt).mul(g4.w.mul(1.2).add(2.8))), 0, 1));
          turbulence.assign(max(turbulence, impact.mul(0.75)));
        }
        state.z.assign(foam);
        state.w.assign(turbulence);
        // how fast the breaking rose since last frame: a wave arriving, which is when surf throws spray
        const surge = max(impact.sub(this.surface.element(i).w), 0).div(u.frameDt);
        this.surface.element(i).assign(vec4(state.x.add(surfHeight), foam, turbulence, impact));
        const flow = this.current[final]!.element(i).xyz.add(windDrift(i));
        this.flow.element(i).assign(vec4(flow, surge));
        // both foam drifts in this vertex's tangent basis, each restarting on its own beat
        const { e1, e2 } = driftBasis(q);
        const moved = vec2(dot(flow, e1), dot(flow, e2)).mul(u.frameDt);
        const drift = this.drift.element(i);
        const reach = (d: Node<'vec2'>) => d.mul(min(float(1), float(DRIFT_REACH).div(max(length(d), 1e-6))));
        drift.assign(
          vec4(reach(drift.xy.add(moved)).mul(u.driftKeep.x), reach(drift.zw.add(moved)).mul(u.driftKeep.y)),
        );
      })().compute(n);

    const finishSlope = (final: number) =>
      Fn(() => {
        const i = instanceIndex;
        const g4 = ground.element(i);
        If(g4.x.lessThanEqual(0), () => {
          this.slope.element(i).assign(vec4(0));
          Return();
        });
        const q = this.vertex.element(i).xyz;
        const here = this.surface.element(i);
        const gradient = vec3(0).toVar();
        Loop({ start: 0, end: 6, type: 'uint' }, ({ i: k }) => {
          const j = this.neighbor.element(i.mul(6).add(k));
          If(j.notEqual(i).and(wetAt(j)), () => {
            const chord = this.vertex.element(j).xyz.sub(q);
            const tangent = chord.sub(q.mul(dot(chord, q)));
            gradient.addAssign(
              tangent.mul(
                this.surface
                  .element(j)
                  .x.sub(here.x)
                  .div(max(dot(tangent, tangent), 1e-6))
                  .div(3),
              ),
            );
          });
        });
        this.slope.element(i).assign(vec4(gradient, 0));

        void final;
      })().compute(n);

    this.stepNodes = [
      [momentum(0), heightStep(0)],
      [momentum(1), heightStep(1)],
    ];
    // Water is kept exactly: once a frame the sea's mean height (area-weighted, in fixed point so it can be
    // summed atomically) is taken back off everywhere. What it corrects is small (pushes fading near the bed,
    // the bed itself, where the shallows hand over to iWave), but on a closed sea it would otherwise add up.
    const volume = instancedArray(1, 'int').toAtomic();
    const volumeScale = VOLUME_FIXED / this.wetArea;
    const clearVolume = Fn(() => {
      atomicStore(volume.element(0), int(0));
    })().compute(1);
    const sumVolume = (final: number) =>
      Fn(() => {
        const i = instanceIndex;
        If(ground.element(i).x.greaterThan(0), () => {
          const share = heights[final]!.element(i).x.mul(this.vertex.element(i).w).mul(volumeScale);
          atomicAdd(volume.element(0), int(round(share)));
        });
      })().compute(n);
    const correctVolume = (final: number) =>
      Fn(() => {
        const i = instanceIndex;
        If(ground.element(i).x.greaterThan(0), () => {
          const mean = float(atomicLoad(volume.element(0))).div(VOLUME_FIXED);
          const state = heights[final]!.element(i);
          state.x.subAssign(mean);
          state.y.subAssign(mean);
        });
      })().compute(n);
    // Splash water: what the fluid took from each vertex as spray or brought back as it landed, raising or
    // lowering it (a landing sends out a small ring), and the foam a landing leaves. Dry land keeps nothing.
    const land = (final: number) =>
      Fn(() => {
        const i = instanceIndex;
        const volume = atomicLoad(this.landings.element(i.mul(2))).toVar();
        const foam = atomicLoad(this.landings.element(i.mul(2).add(1))).toVar();
        atomicStore(this.landings.element(i.mul(2)), int(0));
        atomicStore(this.landings.element(i.mul(2).add(1)), int(0));
        If(ground.element(i).x.lessThanEqual(0), () => {
          Return();
        });
        const state = heights[final]!.element(i);
        const landed = float(foam).div(FLUID.landing.foam);
        state.x.addAssign(float(volume).div(FLUID.landing.volume).div(this.vertex.element(i).w));
        state.z.assign(clamp(state.z.add(landed), 0, 1));
        state.w.assign(clamp(state.w.add(landed.mul(0.35)), 0, 1));
      })().compute(n);
    this.finishNodes = [
      [land(0), clearVolume, sumVolume(0), correctVolume(0), finishSurface(0), finishSlope(0)],
      [land(1), clearVolume, sumVolume(1), correctVolume(1), finishSurface(1), finishSlope(1)],
    ];
    this.resetNodes = [
      Fn(() => {
        const i = instanceIndex;
        for (const buffer of [heights[0]!, heights[1]!, this.surface, this.slope, this.drift])
          buffer.element(i).assign(vec4(0));
      })().compute(n),
      Fn(() => {
        const i = instanceIndex;
        atomicStore(this.landings.element(i.mul(2)), int(0));
        atomicStore(this.landings.element(i.mul(2).add(1)), int(0));
        this.current[0]!.element(i).assign(vec4(0));
        this.current[1]!.element(i).assign(vec4(0));
        this.flow.element(i).assign(vec4(windDrift(i), 0));
      })().compute(n),
    ];
  }

  /**
   * Read-only views for shaders: surface (height with surf, foam, turbulence, surf breaking), slope, foam drift,
   * the current, ground (depth, 0 on land; shore distance; exposure; foam grain), and what's needed to look up a
   * direction.
   */
  renderBuffers() {
    const view = <T extends 'vec4' | 'uint'>(node: StorageBufferNode<T>, type: T) =>
      storage(node.value, type, node.bufferCount).toReadOnly() as StorageBufferNode<T>;
    return {
      surface: view(this.surface, 'vec4'),
      slope: view(this.slope, 'vec4'),
      drift: view(this.drift, 'vec4'),
      driftBlend: this.driftBlend,
      flow: view(this.flow, 'vec4'),
      ground: view(this.ground, 'vec4'),
      vertex: view(this.vertex, 'vec4'),
      neighbor: view(this.neighbor, 'uint'),
      lookup: view(this.lookup, 'uint'),
    };
  }

  reset(): void {
    this.setPressure(null, 0);
    this.renderer.compute(this.resetNodes);
  }

  /**
   * The swell on the shores follows the wind slowly (swell outlasts the wind that raised it), and its clock only
   * ever moves on by its frequency, so a gust or a new wind level changes the surf's size, never its pace.
   */
  private updateSwell(dt: number): void {
    const ease = (tau: number) => 1 - Math.exp(-dt / tau);
    this.swellAngle += (this.uniforms.windAngle.value - this.swellAngle) * ease(25);
    this.swellSpeed += (this.uniforms.wind.value - this.swellSpeed) * ease(15);
    this.uniforms.surfEnergy.value =
      this.uniforms.amplitude.value * Math.min(1.3, Math.max(0, this.swellSpeed));
    const turn = 2 * Math.PI;
    this.uniforms.surfClock.value = (this.uniforms.surfClock.value + this.swell.omega * dt) % turn;
    this.uniforms.setClock.value =
      (this.uniforms.setClock.value + (this.swell.omega / SURF_SET_WAVES) * dt) % turn;
    const heading = new Vector3();
    SHORE_SITES.forEach((site, i) => {
      windOnOrb(site.center, this.swellAngle, heading).normalize();
      this.swellHeadings[i]!.set(heading.x, heading.y, heading.z, 0);
    });
  }

  /** The live wind (core/oceanWind.ts): where it blows, and how hard. */
  setWind(angle: number, speed: number): void {
    this.uniforms.windAngle.value = angle;
    this.uniforms.wind.value = speed;
  }

  /** A finger holding the water down at dir, for heldSeconds; null when it lifts. */
  setPressure(dir: Vector3 | null, heldSeconds: number): void {
    if (dir) this.uniforms.press.value.set(dir.x, dir.y, dir.z, oceanPressLoad(heldSeconds));
    else this.uniforms.press.value.w = 0;
  }

  update(dt: number, forces: readonly OceanForce[], time = 0): void {
    if (dt <= 0) return;
    const steps = Math.min(MAX_STEPS, Math.max(1, Math.round(dt / STEP)));
    this.uniforms.frameDt.value = dt;
    this.uniforms.dt.value = dt / steps;
    this.uniforms.time.value = time;
    this.updateSwell(dt);
    // a drift restarts as its weight reaches 0: the first at whole periods, the second half a period later
    const phase = this.driftClock;
    this.driftClock = (this.driftClock + dt / DRIFT_PERIOD) % 1;
    const next = this.driftClock;
    const keep = (offset: number) => ((next + offset) % 1 < (phase + offset) % 1 ? 0 : 1);
    this.uniforms.driftKeep.value.set(keep(0), keep(0.5));
    this.driftBlend.value = 1 - Math.abs(2 * next - 1);
    for (let e = 0; e < EVENTS; e++) {
      const force = forces[e];
      this.events[e]!.set(force?.dir.x ?? 0, force?.dir.y ?? 0, force?.dir.z ?? 1, force?.strength ?? 0);
      const motion = force?.motion;
      this.eventShapes[e]!.set(
        motion?.x ?? 0,
        motion?.y ?? 0,
        motion?.z ?? 0,
        motion ? HAND.dragSharpness : HAND.tapSharpness,
      );
    }
    this.uniforms.pushMean.value = this.pushMean(dt);
    const passes: ComputeNode[] = [];
    for (let s = 0; s < steps; s++) {
      passes.push(...this.stepNodes[this.parity]!);
      this.parity = 1 - this.parity;
    }
    passes.push(...this.finishNodes[this.parity]!);
    this.renderer.compute(passes);
  }

  /**
   * The area-weighted mean over the wet sea of this frame's pushes (pushAt without the correction): what the
   * hat leaves over where the grain varies or the coast cuts a footprint.
   */
  private pushMean(frameDt: number): number {
    const { positions, areas, wet, forceGrain } = this.field;
    let total = 0;
    const add = (x: number, y: number, z: number, sharpness: number, gain: number, grain: boolean) => {
      // past x = 14 the footprint is nothing
      const reach = 1 - 14 / sharpness;
      for (let i = 0; i < this.count; i++) {
        const alignment = x * positions[i * 3]! + y * positions[i * 3 + 1]! + z * positions[i * 3 + 2]!;
        if (alignment < reach || !wet[i]) continue;
        total += areas[i]! * gain * hat((1 - alignment) * sharpness) * (grain ? forceGrain[i]! : 1);
      }
    };
    for (let e = 0; e < EVENTS; e++) {
      const event = this.events[e]!;
      if (event.w !== 0)
        add(event.x, event.y, event.z, this.eventShapes[e]!.w, (event.w * HAND.impulse) / frameDt, true);
    }
    const press = this.uniforms.press.value;
    if (press.w !== 0) add(press.x, press.y, press.z, HAND.pressSharpness, -press.w * HAND.press, false);
    return total === 0 ? 0 : total / this.wetArea;
  }
}
