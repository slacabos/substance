import type { Vector3 } from 'three';
import type { Node, UniformNode } from 'three/webgpu';
import { acos, atan, clamp, cos, dot, exp, float, max, min, select, sin, smoothstep, vec3 } from 'three/tsl';
import { OCEAN_DEPTH } from '../core/oceanFFT';
import {
  MAINLAND,
  SHELF,
  SHELF_EDGE,
  SHELF_MIN,
  SHORE_SWEEPS,
  SHORE_WATERLINE,
  SURF_FADE,
  SURF_HEIGHT,
  SURF_REACH,
  SURF_SET_WAVES,
  SURF_TRAINS,
  BREAKER,
} from '../core/oceanShores';

type Float = Node<'float'>;
type Vec3 = Node<'vec3'>;

/** The swell the coast's surf and swash follow (OceanSim updates them; core/oceanShores.ts is the reference). */
export interface SurfInputs {
  /** Each surf train's frequency times time (SURF_TRAINS), and the sets' clock, each added up frame by frame. */
  clocks: UniformNode<'vec3', Vector3>;
  setClock: UniformNode<'float', number>;
  /** The sea's state on the shores: amplitude times the wind, eased. */
  energy: UniformNode<'float', number>;
  omega: number;
  k: number;
}

/** The bearing of unit direction q round the land's centre (as shoreSiteDistance measures it). */
function bearingAt(q: Vec3): Float {
  const { east, north } = MAINLAND;
  return atan(dot(q, vec3(north.x, north.y, north.z)), dot(q, vec3(east.x, east.y, east.z)));
}

/**
 * The swell at a point near the beach (shoreSurfPhase, surfBend and surfSet in core/oceanShores.ts): `distance` out
 * from the shore and `lag` the shelf's delay there; its crests come straight in at the beach. Returns each train's
 * phase (a crest where its sine is 1; SURF_TRAINS), and how big this part of a set is.
 */
export function surfWave(
  q: Vec3,
  distance: Float,
  lag: Float,
  surf: SurfInputs,
): { phases: Float[]; set: Float } {
  const bearing = bearingAt(q);
  const out = max(distance.sub(SHELF_EDGE), 0);
  const clock = [surf.clocks.x, surf.clocks.y, surf.clocks.z];
  const phases = SURF_TRAINS.map((train, t) =>
    lag
      .mul(surf.omega * train.ratio)
      .sub(out.mul(surf.k * train.ratio * train.ratio))
      .add(bearing.mul(train.slant))
      .add(sin(bearing.mul(3).add(train.seed)).mul(0.9))
      .add(sin(bearing.mul(7).add(2.3 * train.seed)).mul(0.5))
      .sub(clock[t]!),
  );
  // sets of bigger waves riding in with the crests, a little later along the beach (a whole set a turn, so it
  // joins up all the way round)
  const inward = lag.mul(surf.omega).sub(out.mul(surf.k));
  const lull = sin(inward.div(SURF_SET_WAVES).add(bearing).sub(surf.setClock)).mul(0.5).add(0.5);
  return { phases, set: lull.mul(lull).mul(0.9).add(0.55) };
}

/**
 * The whole surf at a point (shoreSurfAt and sampleShoreSurf in core/oceanShores.ts): every train at its phase,
 * shaped into a bore inside the bar, as tall as the shelf lets it stand, flattening toward the shoreline, and
 * averaged by share. `strength` is the sea's state times the set.
 */
export function surfAtPoint(
  distance: Float,
  phases: Float[],
  strength: Float,
): { height: Float; breaking: Float } {
  const depth = shelfDepthAt(distance);
  const envelope = float(1)
    .sub(smooth(distance.sub(SURF_FADE).div(SURF_REACH - SURF_FADE)))
    .mul(min(float(OCEAN_DEPTH).div(depth).pow(0.25), 1.8))
    .mul(smooth(distance.div(0.025)).mul(0.35).add(0.65));
  const amplitude = min(envelope.mul(strength).mul(SURF_HEIGHT), depth.mul(BREAKER));
  const calm = smooth(distance.sub(0.002).div(0.02));
  const skew = float(1)
    .sub(smooth(distance.sub(SHELF.bar.at + 0.02).div(0.08)))
    .mul(0.8);
  const zone = surfBreakZoneAt(distance);
  let height: Float = float(0);
  let breaking: Float = float(0);
  const shares = SURF_TRAINS.reduce((sum, t) => sum + t.share, 0);
  SURF_TRAINS.forEach((train, t) => {
    const phase = phases[t]!;
    const s = sin(phase.add(skew.mul(sin(phase))));
    const crest = max(s, 0).pow(5);
    const wave = s.mul(0.32).add(crest.mul(0.95)).sub(0.17).mul(amplitude);
    height = height.add(
      wave.mul(select(wave.lessThan(0), calm.mul(0.6).add(0.4), float(1))).mul(train.share),
    );
    // (it breaks over the whole of its steep front, not just at the peak, as in sampleShoreSurf)
    breaking = max(
      breaking,
      max(s, 0)
        .pow(2)
        .mul(zone)
        .mul(strength)
        .mul(Math.min(1, train.share * 1.4)),
    ) as Float;
  });
  const inside = distance.greaterThanEqual(0).and(distance.lessThanEqual(SURF_REACH));
  const total = max(height.mul(1.35 / shares), depth.mul(-BREAKER));
  return { height: select(inside, total, float(0)), breaking: select(inside, breaking, float(0)) };
}

const smooth = (x: Float) => smoothstep(0, 1, x);
/** exp(-x^2), the bar's profile across it (WGSL's pow is undefined for a negative x). */
const bell = (x: Float) => exp(x.mul(x).negate());

/** Still water over the shelf at distance d out (shelfDepth and shelfDrop in core/oceanShores.ts). */
export function shelfDepthAt(d: Float): Float {
  const { width, power, bar } = SHELF;
  const out = max(d, 0);
  const slope = min(out.div(width), 1)
    .pow(power)
    .mul(OCEAN_DEPTH + SHORE_WATERLINE);
  const ridge = bell(out.sub(bar.at).div(bar.width)).mul(bar.height);
  const drop = slope.sub(ridge).add(smooth(out.sub(width).div(0.15)).mul(0.035));
  return clamp(drop.sub(SHORE_WATERLINE), SHELF_MIN, OCEAN_DEPTH);
}

/** Where the surf plunges at distance d out: on the bar and at the beach (surfBreakZone without its whitewater). */
export function surfPlungeAt(d: Float): Float {
  const { bar } = SHELF;
  const beach = float(1).sub(smooth(d.sub(0.012).div(0.06)));
  return max(beach, bell(d.sub(bar.at).div(bar.width)).mul(0.9));
}

/** Where the surf breaks at distance d out (surfBreakZone in core/oceanShores.ts). */
export function surfBreakZoneAt(d: Float): Float {
  const inside = float(1)
    .sub(smooth(d.sub(SHELF.bar.at).div(0.04)))
    .mul(0.45);
  return max(surfPlungeAt(d), inside);
}

/** Angular distance from the beach's shoreline at unit direction d (shoreSiteDistance for MAINLAND). */
export function shoreDistanceAt(d: Vec3): Float {
  const { center, east, north, radius, seed } = MAINLAND;
  const c = vec3(center.x, center.y, center.z);
  const bearing = atan(dot(d, vec3(north.x, north.y, north.z)), dot(d, vec3(east.x, east.y, east.z)));
  let r: Float = float(1);
  for (const s of SHORE_SWEEPS)
    r = r.add(cos(bearing.mul(s.n).add(s.phase + seed * s.seedScale)).mul(s.amplitude));
  return acos(clamp(dot(c, d), -1, 1)).sub(r.mul(radius));
}
