import type { Node, UniformNode } from 'three/webgpu';
import { acos, atan, clamp, cos, dot, exp, float, max, min, sin, smoothstep, vec3 } from 'three/tsl';
import { OCEAN_DEPTH } from '../core/oceanFFT';
import {
  MAINLAND,
  SHELF,
  SHELF_EDGE,
  SHORE_SWEEPS,
  SURF_SLANT,
  SHORE_WATERLINE,
  SURF_SET_WAVES,
} from '../core/oceanShores';

type Float = Node<'float'>;
type Vec3 = Node<'vec3'>;

/** The swell the coast's surf and swash follow (OceanSim updates them; core/oceanShores.ts is the reference). */
export interface SurfInputs {
  /** The swell's frequency times time, and the sets' clock, each added up frame by frame. */
  clock: UniformNode<'float', number>;
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
 * The swell at a point near the beach (shoreSurfPhase and surfSet in core/oceanShores.ts): `distance` out from
 * the shore and `lag` the shelf's delay there; its crests come straight in at the beach. Returns the wave's
 * phase (a crest where its sine is 1) and how big this part of a set is.
 */
export function surfWave(
  q: Vec3,
  distance: Float,
  lag: Float,
  surf: SurfInputs,
): { phase: Float; set: Float } {
  const bearing = bearingAt(q);
  const inward = lag.mul(surf.omega).sub(max(distance.sub(SHELF_EDGE), 0).mul(surf.k));
  // sets of bigger waves riding in with the crests, a little later along the beach (a whole set a turn, so it
  // joins up all the way round)
  const lull = sin(inward.div(SURF_SET_WAVES).add(bearing).sub(surf.setClock)).mul(0.5).add(0.5);
  return {
    phase: inward.add(bearing.mul(SURF_SLANT)).sub(surf.clock),
    set: lull.mul(lull).mul(0.9).add(0.55),
  };
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
  return clamp(drop.sub(SHORE_WATERLINE), 0.002, OCEAN_DEPTH);
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
