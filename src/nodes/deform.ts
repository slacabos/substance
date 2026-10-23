import { Vector3, type Color, type Vector4 } from 'three';
import type { Node, UniformArrayNode, UniformNode } from 'three/webgpu';
import {
  abs,
  acos,
  attribute,
  clamp,
  Continue,
  cos,
  cross,
  dot,
  exp,
  float,
  If,
  Loop,
  max,
  normalize,
  select,
  sin,
  smoothstep,
  vec3,
} from 'three/tsl';
import { MAX_IMPULSES } from '../config';
import { OCEAN_REFERENCE_WAVELENGTH, OCEAN_SPECTRUM } from '../core/oceanSpectrum';
import { OCEAN_WEATHER_PATCHES } from '../core/oceanWeather';
import { snoise } from './noise';
import { baseR } from './shape';

// GPU displacement for the water-like materials. Every surface point is a unit direction pushed out by
// baseR (the orb/cube shape) plus a height from ripples, dents, swells or wobble.

/** Which displacement a deform material uses. */
export const DeformMode = { Water: 0, Slime: 1, Ocean: 2 } as const;
export type DeformMode = (typeof DeformMode)[keyof typeof DeformMode];

export interface DeformUniforms {
  uTime: UniformNode<'float', number>;
  /** Water/ocean: xyz direction, w start time. Slime: xyz direction, w depth. */
  uImp: UniformArrayNode<'vec4'>;
  /** Water/ocean: x strength. Slime: x radius. */
  uImpB: UniformArrayNode<'vec4'>;
  /** The arrays behind uImp and uImpB: write to these, they upload every frame. */
  imp: Vector4[];
  impB: Vector4[];
  /** xyz direction, w how hard the finger is pressing. */
  uPress: UniformNode<'vec4', Vector4>;
  /** xyz axis, w amount of the squash-and-stretch mode. */
  uWob: UniformNode<'vec4', Vector4>;
  /** xyz patch centre, w intensity; matching x radius in radians. */
  uWeather: UniformArrayNode<'vec4'>;
  uWeatherB: UniformArrayNode<'vec4'>;
  weather: Vector4[];
  weatherB: Vector4[];
  uDeep: UniformNode<'color', Color>;
  uShallow: UniformNode<'color', Color>;
  uOceanWind: UniformNode<'float', number>;
  uOceanWavelength: UniformNode<'float', number>;
  uOceanAmplitude: UniformNode<'float', number>;
  uOceanChoppiness: UniformNode<'float', number>;
  uOceanFoam: UniformNode<'float', number>;
}

type Vec3 = Node<'vec3'>;
type Float = Node<'float'>;

const arcDist = (a: Vec3, b: Vec3) => acos(clamp(dot(a, b), -1, 1));
const unit = (x: number, y: number, z: number) => vec3(new Vector3(x, y, z).normalize());

/** Smooth local weather on the sphere; dot products avoid a pole or longitude seam. */
function oceanWeatherAt(p: Vec3, u: DeformUniforms): Float {
  let weather: Float = float(0);
  for (let i = 0; i < OCEAN_WEATHER_PATCHES; i++) {
    const patch = u.uWeather.element(i);
    const radius = u.uWeatherB.element(i).x;
    const inside = smoothstep(cos(radius), cos(radius.mul(0.4)), dot(p, patch.xyz));
    weather = max(weather, inside.mul(patch.w));
  }
  return weather;
}

function oceanAt(
  p: Vec3,
  u: DeformUniforms,
  weather: Float,
): { height: Float; tangent: Vec3; breaking: Float } {
  let height: Float = float(0);
  let tangent: Vec3 = vec3(0);
  let breaking: Float = float(0);
  const warp = sin(
    dot(p, vec3(0.34, 0.87, -0.21))
      .mul(3.2)
      .sub(u.uTime.mul(0.17).mul(u.uOceanWind)),
  )
    .mul(0.32)
    .add(
      sin(
        dot(p, vec3(-0.62, 0.14, 0.77))
          .mul(5.1)
          .add(u.uTime.mul(0.11).mul(u.uOceanWind)),
      ).mul(0.15),
    );
  for (const [amp, [x, y, z], freq, speed, chop, breakWeight] of OCEAN_SPECTRUM) {
    const wind = unit(x, y, z);
    const phase = dot(p, wind)
      .mul(freq)
      .mul(OCEAN_REFERENCE_WAVELENGTH)
      .div(u.uOceanWavelength)
      .add(warp)
      .sub(u.uTime.mul(speed).mul(u.uOceanWind));
    const s = sin(phase);
    const c = cos(phase);
    const a = float(amp)
      .mul(u.uOceanAmplitude)
      .mul(float(1).add(weather.mul(freq < 27 ? 0.2 : 0.75)));
    height = height.add(s.mul(a));
    tangent = tangent.add(wind.sub(p.mul(dot(p, wind))).mul(c.mul(a).mul(chop).mul(u.uOceanChoppiness)));

    const front = smoothstep(0.48, 0.86, s).mul(smoothstep(0.06, 0.72, c.negate()));
    const trail = smoothstep(0.35, 0.88, s)
      .mul(smoothstep(0.12, 0.85, c))
      .mul(0.3);
    breaking = max(breaking, front.add(trail).mul(breakWeight).mul(u.uOceanChoppiness));
  }
  return { height, tangent, breaking };
}

/** Expanding, decaying rings from each tap or drag step, plus the dip under a pressing finger. */
function ripplesAt(p: Vec3, u: DeformUniforms, d: Float): void {
  const t = u.uTime;
  Loop(MAX_IMPULSES, ({ i }) => {
    const a = u.uImp.element(i);
    const str = u.uImpB.element(i).x;
    const age = t.sub(a.w);
    If(str.lessThanEqual(0).or(age.lessThan(0)).or(age.greaterThan(5)), () => {
      Continue();
    });
    const s = arcDist(p, a.xyz);
    const behind = age.mul(0.85).sub(s); // distance behind the wavefront
    const env = smoothstep(-0.06, 0.04, behind)
      .mul(exp(max(behind, 0).mul(-3.2)))
      .mul(exp(age.mul(-0.75)))
      .div(s.mul(1.5).add(1));
    d.addAssign(
      str
        .mul(0.045)
        .mul(env)
        .mul(cos(behind.mul(17))),
    );
    d.subAssign(
      str
        .mul(0.05)
        .mul(exp(s.mul(s).mul(-90)))
        .mul(exp(age.mul(-5))),
    ); // the initial splash dip
  });
  const ps = arcDist(p, u.uPress.xyz);
  d.subAssign(u.uPress.w.mul(0.035).mul(exp(ps.mul(ps).mul(-70))));
}

/** Slow noise, spring-driven dents (each shoving goo aside) and a squash-and-stretch mode. */
function slimeAt(p: Vec3, u: DeformUniforms, d: Float): void {
  const t = u.uTime;
  d.addAssign(snoise(p.mul(2.1).add(vec3(0, t.mul(0.18), t.mul(0.07)))).mul(0.01));
  d.addAssign(snoise(p.mul(5.3).sub(vec3(t.mul(0.12)))).mul(0.003));
  Loop(MAX_IMPULSES, ({ i }) => {
    const a = u.uImp.element(i);
    If(abs(a.w).lessThan(1e-4), () => {
      Continue();
    });
    const r = u.uImpB.element(i).x;
    const s = arcDist(p, a.xyz);
    const rim = s.sub(r.mul(1.7)).div(r.mul(0.75));
    d.addAssign(a.w.mul(exp(s.mul(s).div(r.mul(r)).negate()).sub(exp(rim.mul(rim).negate()).mul(0.32))));
  });
  const c = dot(p, u.uWob.xyz);
  d.addAssign(u.uWob.w.mul(c.mul(c).mul(1.5).sub(0.5)));
}

function dispAt(p: Vec3, mode: DeformMode, u: DeformUniforms): Float {
  const d = float(0).toVar();
  if (mode === DeformMode.Slime) {
    slimeAt(p, u, d);
    return d;
  }
  if (mode === DeformMode.Water) {
    // gentle swell so the water never looks frozen
    const t = u.uTime;
    d.addAssign(
      sin(p.x.mul(5).add(t.mul(1.3)))
        .mul(sin(p.y.mul(4).sub(t.mul(1.1))))
        .mul(0.006),
    );
    d.addAssign(sin(p.z.mul(7).add(p.x.mul(3)).add(t.mul(1.7))).mul(0.004));
  }
  ripplesAt(p, u, d);
  return d;
}

/**
 * Displaced position plus a finite-difference normal, so any displacement shades correctly. Must be called
 * inside a TSL Fn (it emits loops).
 */
export function orbDeform(
  p: Vec3,
  mode: DeformMode,
  u: DeformUniforms,
  coastal = false,
): { pos: Vec3; nrm: Vec3; height: Float; breaking: Float; weather: Float } {
  const weather = mode === DeformMode.Ocean ? oceanWeatherAt(p, u).toVar() : float(0);
  const shoreData = coastal ? attribute('oceanShoreData', 'vec4') : null;
  const shoreDistance = shoreData?.w;
  const shoreNormal = shoreData?.xyz;
  const at = (q: Vec3): { pos: Vec3; height: Float; breaking: Float } => {
    if (mode === DeformMode.Ocean) {
      const wave = oceanAt(q, u, weather);
      const offshore =
        shoreDistance && shoreNormal
          ? smoothstep(0, 0.1, shoreDistance.add(dot(q.sub(p), shoreNormal)))
          : float(1);
      const height = wave.height.mul(offshore.mul(0.82).add(0.18));
      const direction = normalize(q.add(wave.tangent.mul(offshore)));
      return { pos: direction.mul(baseR(direction).add(height)), height, breaking: wave.breaking };
    }
    const height = dispAt(q, mode, u);
    return { pos: q.mul(baseR(q).add(height)), height, breaking: float(0) };
  };
  const surface = at(p);
  const pos = surface.pos.toVar();
  const t1 = normalize(cross(p, select(abs(p.y).lessThan(0.99), vec3(0, 1, 0), vec3(1, 0, 0))));
  const t2 = cross(p, t1);
  const e = 0.004;
  const pa = normalize(p.add(t1.mul(e)));
  const pb = normalize(p.add(t2.mul(e)));
  const A = at(pa).pos;
  const B = at(pb).pos;
  const n = normalize(cross(A.sub(pos), B.sub(pos)));
  const nrm = select(dot(n, p).lessThan(0), n.negate(), n);
  return { pos, nrm, height: surface.height, breaking: surface.breaking, weather };
}
