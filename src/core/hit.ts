import { Vector3 } from 'three';

/** Half-size of the cube shape's flat faces (matches baseR() in shape.glsl). */
export const CUBE_HALF = 0.8;

/**
 * Intersects a ray, given in the object's local space, with the current shape.
 * The shape is treated as a sphere while mostly round and as a box once mostly cube.
 * Returns the hit as a unit direction from the centre, which is what the shaders use to place effects.
 * With `clampToEdge`, a miss snaps to the nearest point of the silhouette so drags can slide off the edge.
 */
export function intersectShape(
  origin: Vector3,
  dir: Vector3,
  morph: number,
  padding: number,
  clampToEdge: boolean,
): Vector3 | null {
  let t: number | null =
    morph < 0.5 ? raySphere(origin, dir, 1 + padding) : rayBox(origin, dir, CUBE_HALF + padding);
  if (t === null) {
    if (!clampToEdge) return null;
    t = -origin.dot(dir);
  }
  const hit = dir.clone().multiplyScalar(t).add(origin);
  if (hit.lengthSq() < 1e-12) return null;
  return hit.normalize();
}

export function raySphere(origin: Vector3, dir: Vector3, radius: number): number | null {
  const b = origin.dot(dir);
  const disc = b * b - (origin.lengthSq() - radius * radius);
  if (disc < 0) return null;
  const t = -b - Math.sqrt(disc);
  return t >= 0 ? t : null;
}

export function rayBox(origin: Vector3, dir: Vector3, half: number): number | null {
  let tmin = -Infinity;
  let tmax = Infinity;
  for (const k of ['x', 'y', 'z'] as const) {
    const o = origin[k];
    const d = dir[k];
    if (Math.abs(d) < 1e-9) {
      if (Math.abs(o) > half) return null;
      continue;
    }
    const t1 = (-half - o) / d;
    const t2 = (half - o) / d;
    tmin = Math.max(tmin, Math.min(t1, t2));
    tmax = Math.min(tmax, Math.max(t1, t2));
  }
  if (tmin > tmax || tmin < 0) return null;
  return tmin;
}
