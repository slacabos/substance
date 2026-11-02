import type { Node } from 'three/webgpu';
import {
  abs,
  cos,
  cross,
  dot,
  float,
  hash,
  instanceIndex,
  normalize,
  select,
  sin,
  smoothstep,
  sqrt,
  vec3,
} from 'three/tsl';
import { PLANE } from '../core/chart';
import { shapePlaneWeight } from './shape';

// On the plane each drawn strand covers far more ground than on the orb, so fur and fluff draw every simulated
// strand several times over there (FUR.plane.copies, FLUFF.plane.copies). Instances run copy by copy: the first
// `count` are the strands themselves, which the orb and the cube draw alone, and each later run is one more copy
// of them, its root scattered across its strand's patch and its shape turned its own way about the root, so the
// copies don't read as clones. Nothing extra is simulated: every copy moves as its strand does. The copies thin
// out as the plane goes, gone by COPIES_FROM of the way, so the orb never wears them while it takes shape.

/** The plane's weight below which no copies are drawn; they thin out from COPIES_FULL down to it. */
export const COPIES_FROM = 0.3;
const COPIES_FULL = 0.8;

/** Dispatch copies conservatively before the nearest roots reach their visibility threshold.
 * Their individual shader weights hide them until needed, including during a reversed morph.
 */
export function strandsDrawn(count: number, copies: number, plane: number): number {
  return count * (plane > COPIES_FROM * PLANE.unfold.duration ? copies : 1);
}

type Vec3 = Node<'vec3'>;
type Float = Node<'float'>;

export interface StrandCopy {
  /** The simulated strand this instance draws. */
  readonly strand: Node<'uint'>;
  /** How much of this copy shows, for its width: 1 for the strand itself, thinning out as the plane goes. */
  shown(p: Vec3): Float;
  /** The copy's root: the strand's own (unit direction p), or scattered across its patch for later copies. */
  root(p: Vec3): Vec3;
  /** Turns a vector off the root (n its unit normal) the copy's own way: unchanged for the strand itself. */
  turn(v: Vec3, n: Vec3): Vec3;
}

/** The strand an instance draws among `count`, and how its copy differs; `spacing` is the strands' own (radians). */
export function strandCopy(count: number, spacing: number): StrandCopy {
  const strand = instanceIndex.mod(count);
  const copy = select(instanceIndex.lessThan(count), float(0), float(1));
  const angle = hash(instanceIndex.add(count * 7)).mul(Math.PI * 2);
  const reach = sqrt(hash(instanceIndex.add(count * 11))).mul(spacing * 0.6);
  const twist = hash(instanceIndex.add(count * 13))
    .mul(Math.PI * 2)
    .mul(copy);
  return {
    strand,
    shown: (p) =>
      float(1).sub(copy.mul(float(1).sub(smoothstep(COPIES_FROM, COPIES_FULL, shapePlaneWeight(p))))),
    root: (p: Vec3): Vec3 => {
      const e1 = normalize(cross(p, select(abs(p.y).lessThan(0.9), vec3(0, 1, 0), vec3(1, 0, 0))));
      const e2 = cross(p, e1);
      const offset = e1
        .mul(cos(angle))
        .add(e2.mul(sin(angle)))
        .mul(reach.mul(copy));
      return normalize(p.add(offset));
    },
    turn: (v: Vec3, n: Vec3): Vec3 => {
      // Rodrigues' rotation by the copy's twist about n
      const c: Float = cos(twist);
      const s: Float = sin(twist);
      return v
        .mul(c)
        .add(cross(n, v).mul(s))
        .add(n.mul(dot(n, v).mul(float(1).sub(c))));
    },
  };
}

/** The spacing between `count` strands spread evenly over the orb, radians. */
export function strandSpacing(count: number): number {
  return Math.sqrt((4 * Math.PI) / count);
}
