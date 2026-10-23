import type { Node } from 'three/webgpu';
import {
  abs,
  clamp,
  cross,
  dot,
  float,
  floor,
  fract,
  max,
  mix,
  normalize,
  normalViewGeometry,
  positionView,
  positionViewDirection,
  pow,
  sign,
  sin,
  smoothstep,
  vec3,
} from 'three/tsl';
import type { DeformUniforms } from './deform';
import { OCEAN_REFERENCE_WAVELENGTH } from '../core/oceanSpectrum';

type Vec3 = Node<'vec3'>;
type Float = Node<'float'>;

function h31(x: Vec3): Float {
  const p = fract(x.mul(0.3183099).add(0.1)).mul(17);
  return fract(p.x.mul(p.y).mul(p.z).mul(p.x.add(p.y).add(p.z)));
}

function vnoise(x: Vec3): Float {
  const i = floor(x);
  const f0 = fract(x);
  const f = f0.mul(f0).mul(f0.mul(-2).add(3));
  const c = (dx: number, dy: number, dz: number) => h31(i.add(vec3(dx, dy, dz)));
  return mix(
    mix(mix(c(0, 0, 0), c(1, 0, 0), f.x), mix(c(0, 1, 0), c(1, 1, 0), f.x), f.y),
    mix(mix(c(0, 0, 1), c(1, 0, 1), f.x), mix(c(0, 1, 1), c(1, 1, 1), f.x), f.y),
    f.z,
  );
}

/** Whitecaps follow steep travelling crests; simulated foam persists and drifts after impacts. */
export function oceanFoam(
  obj: Vec3,
  breaking: Float,
  weather: Float,
  simulated: Float,
  u: DeformUniforms,
): Float {
  // Narrow in the swell direction and stretched along each crest.
  const scale = float(OCEAN_REFERENCE_WAVELENGTH).div(u.uOceanWavelength);
  const q = vec3(
    dot(obj, vec3(0.35, 0.9, 0.25))
      .mul(79)
      .mul(scale)
      .sub(u.uTime.mul(1.3).mul(u.uOceanWind)),
    dot(obj, vec3(-0.88, 0.32, 0.34))
      .mul(17)
      .mul(scale)
      .add(u.uTime.mul(0.18).mul(u.uOceanWind)),
    dot(obj, vec3(0.3, -0.16, 0.94))
      .mul(12)
      .mul(scale),
  );
  const n = vnoise(q)
    .mul(0.7)
    .add(vnoise(q.mul(2.35)).mul(0.3));
  // Keep the transition narrow while retaining derivative antialiasing at small screen sizes.
  const threshold = mix(0.655, 0.52, weather);
  const width = max(0.018, n.fwidth().mul(0.5));
  const coverage = smoothstep(threshold.sub(width), threshold.add(width), n);
  const crests = smoothstep(0.16, 0.78, breaking)
    .mul(coverage)
    .mul(mix(0.2, 0.76, smoothstep(0.14, 0.65, weather)));
  const sprayFoam = smoothstep(0.09, 0.6, simulated).mul(smoothstep(0.36, 0.72, n));
  return clamp(max(crests, sprayFoam).mul(u.uOceanFoam), 0, 1);
}

/** Approximate absorption through a deep body of water, plus a blue grazing-angle glint. */
export function oceanColor(wave: Float, foam: Float, weather: Float, u: DeformUniforms): Vec3 {
  const facing = clamp(dot(normalViewGeometry, positionViewDirection), 0, 1);
  const depth = smoothstep(0.2, 0.9, facing);
  const tint = smoothstep(-0.025, 0.025, wave).mul(0.35).add(depth.mul(0.1));
  const water = mix(u.uDeep, u.uShallow, tint).mul(float(1).sub(weather.mul(0.05)));
  const rim = pow(float(1).sub(facing), 5).mul(0.12);
  const lit = mix(water, vec3(0.08, 0.29, 0.39), rim);
  return mix(lit, vec3(0.93, 0.97, 0.965), foam);
}

/** Subpixel waves change highlights without adding more geometry or aliasing at small sizes. */
export function oceanFineNormal(obj: Vec3, turbulence: Float, u: DeformUniforms): Vec3 {
  const scale = float(OCEAN_REFERENCE_WAVELENGTH).div(u.uOceanWavelength);
  const phaseA = dot(obj, vec3(0.91, 0.2, 0.36))
    .mul(61)
    .mul(scale)
    .sub(u.uTime.mul(3.1).mul(u.uOceanWind));
  const phaseB = dot(obj, vec3(0.32, -0.54, 0.78))
    .mul(103)
    .mul(scale)
    .sub(u.uTime.mul(4.1).mul(u.uOceanWind));
  const fadeA = float(1).sub(smoothstep(0.55, 1.5, phaseA.fwidth()));
  const fadeB = float(1).sub(smoothstep(0.55, 1.5, phaseB.fwidth()));
  const activity = float(1).add(turbulence.mul(0.8));
  const h = sin(phaseA)
    .mul(fadeA)
    .mul(0.00085)
    .add(sin(phaseB).mul(fadeB).mul(0.00045))
    .mul(activity)
    .mul(u.uOceanAmplitude);

  const dx = positionView.dFdx();
  const dy = positionView.dFdy();
  const n = normalViewGeometry;
  const r1 = cross(dy, n);
  const r2 = cross(n, dx);
  const det = dot(dx, r1);
  const gradient = r1.mul(h.dFdx()).add(r2.mul(h.dFdy()));
  return normalize(n.mul(abs(det)).sub(gradient.mul(sign(det))));
}
