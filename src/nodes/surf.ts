import type { Node, UniformNode } from 'three/webgpu';
import { acos, clamp, dot, exp, float, max, min, normalize, sin, smoothstep, vec3 } from 'three/tsl';
import type { Vector3 } from 'three';
import { OCEAN_DEPTH } from '../core/oceanFFT';
import { SHELF, SHELF_EDGE, SHORE_WATERLINE, SURF_SET_WAVES } from '../core/oceanShores';

type Float = Node<'float'>;
type Vec3 = Node<'vec3'>;

/** The swell the coast's surf and swash follow (OceanSim updates them; core/oceanShores.ts is the reference). */
export interface SurfInputs {
  /** The point on the orb the swell's crests close in on. */
  target: UniformNode<'vec3', Vector3>;
  /** The swell's frequency times time, and the sets' clock, each added up frame by frame. */
  clock: UniformNode<'float', number>;
  setClock: UniformNode<'float', number>;
  /** The sea's state on the shores: amplitude times the wind, eased. */
  energy: UniformNode<'float', number>;
  omega: number;
  k: number;
}

/**
 * The swell at a point near the coast (shoreSurfPhase and surfSet in core/oceanShores.ts): `distance` out from
 * the shore, `lag` the shelf's delay there, `site` the nearest island, `normal` pointing out to sea. Returns the
 * wave's phase (a crest where its sine is 1), how big this part of a set is, and how squarely the shore faces
 * the swell, as a 0.15 to 1 exposure.
 */
export function surfWave(
  q: Vec3,
  distance: Float,
  lag: Float,
  site: Float,
  normal: Vec3,
  surf: SurfInputs,
): { phase: Float; set: Float; exposure: Float } {
  const toward = dot(q, surf.target);
  const heading = normalize(surf.target.sub(q.mul(toward)).add(q.cross(vec3(1, 0, 0)).mul(1e-4)));
  const facing = dot(heading, normal.negate());
  const across = max(float(SHELF_EDGE).sub(distance), 0);
  // how far the swell has come: the arc still to run to its target, counted down
  const spatial = acos(clamp(toward, -1, 1))
    .negate()
    .mul(surf.k)
    .add(lag.mul(surf.omega))
    .sub(across.mul(surf.k).mul(max(facing, 0.15)));
  // sets of bigger waves riding in with the crests, at each island's own times
  const lull = sin(spatial.div(SURF_SET_WAVES).sub(surf.setClock).add(site.mul(2.4)))
    .mul(0.5)
    .add(0.5);
  return {
    phase: spatial.sub(surf.clock),
    set: lull.mul(lull).mul(0.9).add(0.55),
    exposure: smoothstep(0, 1, facing.add(0.2).div(0.8)).mul(0.85).add(0.15),
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

/** Where the surf breaks at distance d out (surfBreakZone in core/oceanShores.ts). */
export function surfBreakZoneAt(d: Float): Float {
  const { bar } = SHELF;
  const beach = float(1).sub(smooth(d.sub(0.012).div(0.06)));
  const onBar = bell(d.sub(bar.at).div(bar.width)).mul(0.9);
  const inside = float(1)
    .sub(smooth(d.sub(bar.at).div(0.04)))
    .mul(0.45);
  return max(beach, max(onBar, inside));
}
