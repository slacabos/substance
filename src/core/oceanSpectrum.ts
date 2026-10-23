import { Vector3 } from 'three';
import { OCEAN_DEFAULTS, type OceanTuning } from './oceanTuning';

/** Wavelength used when the original wave spectrum was authored. */
export const OCEAN_REFERENCE_WAVELENGTH = 0.33;

/** Directional, short-fetch waves. Distances and amplitudes are relative to the orb radius. */
export const OCEAN_SPECTRUM = [
  [0.0055, [0.35, 0.9, 0.25], 13, 1.0, 0.45, 0.8],
  [0.0043, [0.53, 0.8, 0.28], 19, 1.27, 0.52, 0.85],
  [0.0035, [-0.08, 0.94, 0.34], 27, 1.61, 0.48, 1],
  [0.003, [0.69, 0.6, 0.39], 34, 1.92, 0.48, 0.8],
  [0.0022, [0.15, 0.72, 0.68], 43, 2.26, 0.38, 0.7],
  [0.0018, [-0.43, 0.79, 0.44], 52, 2.67, 0.32, 0.6],
  [0.0013, [0.49, 0.53, -0.69], 63, 3.16, 0.25, 0.5],
  [0.001, [-0.33, 0.59, -0.74], 76, 3.63, 0.2, 0.4],
] as const;

const DIRECTIONS = OCEAN_SPECTRUM.map(([, [x, y, z]]) => new Vector3(x, y, z).normalize());
const t = new Vector3();
const q = new Vector3();

/** Matches the shader displacement so surface foam remains attached to the water. */
export function sampleOceanSurface(
  dir: Vector3,
  time: number,
  weather: number,
  morph: number,
  out: Vector3,
  tuning: Readonly<OceanTuning> = OCEAN_DEFAULTS,
): number {
  const wavelengthScale = OCEAN_REFERENCE_WAVELENGTH / tuning.peakWavelength;
  const warp =
    Math.sin((dir.x * 0.34 + dir.y * 0.87 - dir.z * 0.21) * 3.2 - time * 0.17 * tuning.windSpeed) * 0.32 +
    Math.sin((-dir.x * 0.62 + dir.y * 0.14 + dir.z * 0.77) * 5.1 + time * 0.11 * tuning.windSpeed) * 0.15;
  let height = 0;
  out.copy(dir);
  for (let i = 0; i < OCEAN_SPECTRUM.length; i++) {
    const [amplitude, , frequency, speed, chop] = OCEAN_SPECTRUM[i]!;
    const wind = DIRECTIONS[i]!;
    const projection = dir.dot(wind);
    const phase = projection * frequency * wavelengthScale + warp - time * speed * tuning.windSpeed;
    const amp = amplitude * tuning.amplitude * (1 + weather * (frequency < 27 ? 0.2 : 0.75));
    height += Math.sin(phase) * amp;
    t.copy(wind)
      .addScaledVector(dir, -projection)
      .multiplyScalar(Math.cos(phase) * amp * chop * tuning.choppiness);
    out.add(t);
  }
  out.normalize();
  const m = Math.min(1.15, Math.max(-0.15, morph));
  const power = Math.pow(Math.abs(out.x), 10) + Math.pow(Math.abs(out.y), 10) + Math.pow(Math.abs(out.z), 10);
  const radius = 1 + (0.8 / Math.pow(power, 0.1) - 1) * m;
  q.copy(out).multiplyScalar(radius + height);
  out.copy(q);
  return height;
}
