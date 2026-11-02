import { MathUtils, Vector3 } from 'three';
import { PLANE } from './chart';

/** One reversible timeline for opening the solid into ground. */
export const UNFURL = {
  cubeRoundFrom: 0.08,
  cubeRoundUntil: 0.65,
  alignUntil: 0.24,
  cameraFrom: 0.12,
  cameraUntil: 0.88,
  startRadius: 1.45,
  undersideFrom: -0.65,
  undersideUntil: -0.98,
  undersideDepth: 2.5,
  seamFrom: 0.12,
  seamUntil: 0.3,
  seamPole: -0.94,
  bubblesUntil: 0.12,
  endRadius: PLANE.rhoFar + PLANE.footZ + 1,
} as const;

export const unfurlCameraWeight = (progress: number): number =>
  MathUtils.smoothstep(progress, UNFURL.cameraFrom, UNFURL.cameraUntil);

export const unfurlRotationWeight = (progress: number): number =>
  MathUtils.smoothstep(progress, 0, UNFURL.alignUntil);

/** Keep the starting solid recognizable until the opening has begun. */
export const unfurlSolidWeight = (cube: number, progress: number): number =>
  cube * (1 - MathUtils.smoothstep(progress, UNFURL.cubeRoundFrom, UNFURL.cubeRoundUntil));

/** The far ground grows out from the front instead of sweeping across the view as long triangles. */
export function unfurlRadius(progress: number): number {
  const t = MathUtils.smoothstep(progress, 0, 1);
  return MathUtils.lerp(UNFURL.startRadius, UNFURL.endRadius, t * t * t);
}

export function unfurlPoint(
  point: Vector3,
  progress: number,
  direction?: Vector3,
  out = new Vector3(),
): Vector3 {
  const dx = point.x;
  const dz = point.z - 1;
  const scale = Math.min(1, unfurlRadius(progress) / Math.max(Math.hypot(dx, dz), 1e-6));
  const back = direction
    ? 1 - MathUtils.smoothstep(direction.z, UNFURL.undersideUntil, UNFURL.undersideFrom)
    : 0;
  const t = MathUtils.clamp(progress, 0, 1);
  const tuck = UNFURL.undersideDepth * 4 * t * (1 - t) * back;
  return out.set(dx * scale, point.y - tuck, 1 + dz * scale);
}

/** The closed solid opens at its back pole; no invertible visible ground lies in this cap. */
export function unfurlVisible(direction: Vector3, progress: number): boolean {
  const cutoff = MathUtils.lerp(
    -1.001,
    UNFURL.seamPole,
    MathUtils.smoothstep(progress, UNFURL.seamFrom, UNFURL.seamUntil),
  );
  return direction.z > cutoff;
}
