import { MathUtils, Vector3 } from 'three';

// The plane shape lays the orb out flat as endless ground. Everything on the orb lives at a unit direction p, and the
// simulations keep running there; only drawing moves each direction to its point on the ground. The chart is
// conformal, so a round ripple, dent or crater on the orb stays round on the ground, just scaled:
//   0. the orb turned about the axis toward the camera, so its top (the coast's land) comes out to the right;
//   1. stereographic projection about the face toward the camera, C = (0, 0, 1): σ = 2 (p.x, p.y) / (1 + p.z);
//   2. a log-polar fan around the foot of the camera: distance ρ = ρ0 exp(-κ σy), bearing φ = -κ σx.
// The coast's beach comes out on the right, its shoreline running in from the distance on the left down to the
// bottom right, and the sea fills the left and the foreground out to the horizon; the back of the orb folds away
// off screen: under the camera, out to the sides and onto a far ring. Orb right comes out on the left, which keeps
// the ground's outward side up.

/** The chart's constants, in orb radii. Tuned so the mesh stays within a few times the orb's on-screen spacing. */
export const PLANE = {
  /** The turn about the axis toward the camera, radians: the coast's land comes out on the right. */
  turn: MathUtils.degToRad(120),
  /** The ground's height. */
  groundY: -1,
  /** How fast distance grows across the orb: larger reaches nearer and further but uses less of the mesh. */
  kappa: 0.35,
  /** Where C lands, measured from the camera's foot. */
  rho0: 6,
  /** The camera's foot is this far toward the camera from the origin, so C lands at the orb's front, (0, -1, 1). */
  footZ: 7,
  /** The bearing the chart stops at either side, radians: well outside the view. */
  phiMax: 1.3,
  /** The nearest and furthest the chart reaches from the camera's foot. */
  rhoMin: 0.3,
  rhoFar: 150,
  /** The most a height off the surface is scaled up by, so tall features far away stay within reason. */
  heightCap: 5,
  /**
   * The ground's scale where the orb's front lands (kappa * rho0), which things drawn at one size across the ground
   * (hair) take: grown with the chart's own scale instead, they grew with distance faster than perspective
   * shrank them, and the background looked magnified.
   */
  detail: 0.35 * 6,
  /** Distance-based unfolding schedule, shared by the CPU reference and shaders. */
  unfold: { near: 3, far: 60, delay: 0.35, duration: 0.65 },
  /**
   * The fixed camera: its height over the ground, and where the horizon sits on screen, as a fraction of the way
   * from the middle to the top edge. A wide view lets it look down steeply and still leave a band of sky.
   */
  eyeHeight: 2.5,
  horizon: 0.64,
} as const;

const TURN_COS = Math.cos(PLANE.turn);
const TURN_SIN = Math.sin(PLANE.turn);

/** The orb turned about the axis toward the camera (step 0); `back` turns it back. */
const turned = (v: Vector3, back = false): [number, number, number] => {
  const s = back ? -TURN_SIN : TURN_SIN;
  return [v.x * TURN_COS - v.y * s, v.x * s + v.y * TURN_COS, v.z];
};

/** The stereographic coordinates of a unit direction, kept finite at the back pole. */
const sigmaOf = (p: Vector3): [number, number, number] => {
  const [x, y, z] = turned(p);
  const d = Math.max(1 + z, 1e-6);
  return [(2 * x) / d, (2 * y) / d, d];
};

const fanOf = (sx: number, sy: number): [rho: number, phi: number] => [
  MathUtils.clamp(PLANE.rho0 * Math.exp(-PLANE.kappa * sy), PLANE.rhoMin, PLANE.rhoFar),
  MathUtils.clamp(-PLANE.kappa * sx, -PLANE.phiMax, PLANE.phiMax),
];

/** Where the direction p lies on the ground. */
export function planePoint(p: Vector3, out = new Vector3()): Vector3 {
  const [sx, sy] = sigmaOf(p);
  const [rho, phi] = fanOf(sx, sy);
  return out.set(rho * Math.sin(phi), PLANE.groundY, PLANE.footZ - rho * Math.cos(phi));
}

/** How much the ground is scaled up from the orb around p: ground length per unit of direction. */
export function planeScale(p: Vector3): number {
  const [sx, sy] = sigmaOf(p);
  const [rho] = fanOf(sx, sy);
  return (PLANE.kappa * rho * (4 + sx * sx + sy * sy)) / 4;
}

/** The scale capped for heights off the surface (planeShell's h). */
export function planeHeightScale(p: Vector3): number {
  return Math.min(planeScale(p), PLANE.heightCap);
}

/** Where a step t along the orb's surface at p goes on the ground: the chart's differential (inside its clamps). */
export function planePush(p: Vector3, t: Vector3, out = new Vector3()): Vector3 {
  const [sx, sy, d] = sigmaOf(p);
  const [rho, phi] = fanOf(sx, sy);
  const [px, py] = turned(p);
  const [tx, ty, tz] = turned(t);
  const dsx = (2 * (tx * d - px * tz)) / (d * d);
  const dsy = (2 * (ty * d - py * tz)) / (d * d);
  const dRho = -PLANE.kappa * rho * dsy;
  const dPhi = -PLANE.kappa * dsx;
  const sin = Math.sin(phi);
  const cos = Math.cos(phi);
  return out.set(dRho * sin + rho * dPhi * cos, 0, -dRho * cos + rho * dPhi * sin);
}

/**
 * The step along the orb's surface at p that the chart carries along the ground vector g (its scale aside): the
 * transpose of its differential, which for a conformal map points the same way as its inverse.
 */
export function planePull(p: Vector3, g: Vector3, out = new Vector3()): Vector3 {
  const e1 = new Vector3(Math.abs(p.y) < 0.9 ? 0 : 1, Math.abs(p.y) < 0.9 ? 1 : 0, 0)
    .cross(p)
    .negate()
    .normalize();
  const e2 = p.clone().cross(e1);
  return out
    .copy(e1)
    .multiplyScalar(g.dot(planePush(p, e1)))
    .addScaledVector(e2, g.dot(planePush(p, e2)));
}

/** The direction whose ground point is (x, groundY, z): the chart's inverse, clamped into its range. */
export function planeDirection(x: number, z: number, out = new Vector3()): Vector3 {
  const rho = MathUtils.clamp(Math.hypot(x, PLANE.footZ - z), PLANE.rhoMin, PLANE.rhoFar);
  const phi = MathUtils.clamp(Math.atan2(x, PLANE.footZ - z), -PLANE.phiMax, PLANE.phiMax);
  const sx = -phi / PLANE.kappa;
  const sy = -Math.log(rho / PLANE.rho0) / PLANE.kappa;
  const s2 = sx * sx + sy * sy;
  out.set(4 * sx, 4 * sy, 4 - s2).divideScalar(4 + s2);
  const [ux, uy] = turned(out, true);
  return out.set(ux, uy, out.z);
}

/**
 * The shell map: a point x near the orb's surface (whose base surface is `baseRadius` out along its direction) moved
 * onto the ground, its height off the surface scaled with the ground's scale up to the cap.
 */
export function planeShell(x: Vector3, baseRadius = 1, out = new Vector3()): Vector3 {
  const p = x.clone().normalize();
  const height = (x.length() - baseRadius) * planeHeightScale(p);
  return planePoint(p, out).setY(PLANE.groundY + height);
}

/**
 * A surface normal n at direction p, as the shell map turns it: the inverse transpose of the map's Jacobian, which
 * scales the surface's slopes by h / s (heights grow by h, lengths along the ground by s).
 */
export function planeNormal(p: Vector3, n: Vector3, out = new Vector3()): Vector3 {
  const s = planeScale(p);
  const along = n.dot(p);
  const tangent = n.clone().addScaledVector(p, -along);
  planePush(p, tangent, out).multiplyScalar(planeHeightScale(p) / (s * s));
  return out.setY(out.y + along).normalize();
}

export interface PlaneView {
  /** Where the camera stands and the point it looks at. */
  readonly eye: Vector3;
  readonly target: Vector3;
  /** Vertical field of view, degrees: wider on portrait screens, which see the ground through a narrow slot. */
  readonly fov: number;
  /** The direction whose ground point is in the middle of the view. */
  readonly focus: Vector3;
}

/** The plane's fixed camera for a screen of this aspect (width / height). */
export function planeView(aspect: number): PlaneView {
  const fov = MathUtils.lerp(60, 55, MathUtils.clamp((aspect - 0.6) / 0.9, 0, 1));
  const pitch = Math.atan(PLANE.horizon * Math.tan(MathUtils.degToRad(fov / 2)));
  const eye = new Vector3(0, PLANE.groundY + PLANE.eyeHeight, PLANE.footZ);
  const target = new Vector3(0, PLANE.groundY, PLANE.footZ - PLANE.eyeHeight / Math.tan(pitch));
  return { eye, target, fov, focus: planeDirection(target.x, target.z) };
}

/** Fixed ground coordinates in the units used to author the orb's detail. */
export function planeDetailPoint(p: Vector3, out = new Vector3()): Vector3 {
  planePoint(p, out);
  return out.set(out.x / PLANE.detail, 0, (out.z - 1) / PLANE.detail);
}

/** Near ground unfolds first; reversing progress retracts the far ground first. */
export function planeWeight(p: Vector3, progress: number): number {
  const q = planePoint(p);
  const distance = Math.hypot(q.x, PLANE.footZ - q.z);
  const delay = PLANE.unfold.delay * MathUtils.smoothstep(distance, PLANE.unfold.near, PLANE.unfold.far);
  return MathUtils.smoothstep(progress, delay, delay + PLANE.unfold.duration);
}
