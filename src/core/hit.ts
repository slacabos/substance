import { Vector3 } from 'three';
import { PLANE, planeDirection } from './chart';

/** Half-size of the cube shape's flat faces (matches baseR() in nodes/shape.ts). */
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

/**
 * Intersects a ray in object space with the plane shape's ground, returning the direction whose ground point it hits
 * (core/chart.ts). It never misses: a ray at or above the horizon lands on the far ring, along its bearing.
 */
export function intersectPlane(origin: Vector3, dir: Vector3): Vector3 {
  const drop = origin.y - PLANE.groundY;
  const t = dir.y < -1e-6 ? drop / -dir.y : Infinity;
  const x = origin.x + dir.x * t;
  const z = origin.z + dir.z * t;
  if (Number.isFinite(t) && Math.hypot(x, PLANE.footZ - z) < PLANE.rhoFar) return planeDirection(x, z);
  const bearing = Math.hypot(dir.x, dir.z) > 1e-9 ? Math.atan2(dir.x, -dir.z) : 0;
  return planeDirection(PLANE.rhoFar * Math.sin(bearing), PLANE.footZ - PLANE.rhoFar * Math.cos(bearing));
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
