/** Rendering and recovery limits for the plane; orb/cube retain their existing response. */
export const PLANE_RESPONSE = {
  maxStreakAspect: 6,
  extraWaveDamping: 0.45,
  extraCurrentDrag: 0.65,
  weatherRecovery: 2,
} as const;

export function planeWeatherRecovery(plane: number, held: boolean): number {
  return held ? 5 : 5 + (PLANE_RESPONSE.weatherRecovery - 5) * plane;
}

export function planeStreakReach(size: number, speed: number, shutter: number): number {
  return Math.min(size + speed * shutter, size * PLANE_RESPONSE.maxStreakAspect);
}
