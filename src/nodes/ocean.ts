import type { Texture } from 'three';
import type { Node } from 'three/webgpu';
import { LACE_RANGE } from '../core/foamLace';
import {
  cameraViewMatrix,
  abs,
  clamp,
  dot,
  float,
  floor,
  fract,
  fwidth,
  length,
  max,
  mix,
  normalize,
  normalViewGeometry,
  positionViewDirection,
  pow,
  smoothstep,
  sqrt,
  texture,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { DeformUniforms } from './deform';

type Vec3 = Node<'vec3'>;
type Float = Node<'float'>;

/** Wavelength the foam pattern's scale was authored at. */
const FOAM_REFERENCE_WAVELENGTH = 0.33;

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

/** Where the foam pattern is read: two drifted positions and how much of it follows the first (see OceanSim). */
export interface FoamDrift {
  a: Vec3;
  b: Vec3;
  blend: Float;
}

/** Foam lace cells per orb radius at the authored wavelength: bubbles a few pixels across. */
const LACE = 40;
/** Cells across the baked lace tile (core/foamLace.ts). */
export const LACE_TILE_CELLS = 32;

/**
 * The baked bubbles at obj (core/foamLace.ts): x the distance to the nearest bubble's centre, in cell widths; y
 * that bubble's own random number. Projected along the three axes and blended by how squarely each faces the
 * surface; each bubble size is offset so they never line up.
 */
function bubblesAt(lace: Texture, obj: Vec3, frequency: number, scale: Float): Node<'vec2'> {
  const n = normalize(obj);
  const w = pow(abs(n), vec3(6));
  const weights = w.div(w.x.add(w.y).add(w.z));
  const at = obj.mul(scale.mul(frequency / LACE_TILE_CELLS)).add(frequency * 0.37);
  const read = (uv: Node<'vec2'>) => texture(lace, uv).xy;
  const sample = read(at.yz).mul(weights.x).add(read(at.zx).mul(weights.y)).add(read(at.xy).mul(weights.z));
  return vec2(sample.x.mul(LACE_RANGE), sample.y);
}

/**
 * Churned foam: white water full of round holes. Where the sim has foam, a slow noise gathers it into clumps
 * with open water between. Within them each bubble opens to its own radius: pinholes in thick foam, growing and
 * merging into a thin, translucent web as it thins, and a few large bubbles open wide. Each bubble size is read
 * at both drifts and crossfaded before the holes are cut, so bubbles never double up. Already scaled by the
 * foam strength.
 */
function foamLace(drift: FoamDrift, lace: Texture, amount: Float, u: DeformUniforms): Float {
  const scale = float(FOAM_REFERENCE_WAVELENGTH).div(u.uOceanWavelength).mul(LACE);
  const middle = mix(drift.b, drift.a, drift.blend).mul(scale);
  const clumps = vnoise(middle.mul(0.08))
    .mul(0.65)
    .add(vnoise(middle.mul(0.21).add(7.1)).mul(0.35));
  const held = clamp(amount.mul(clumps.sub(0.3).mul(2.6)), 0, 1);
  const thin = float(1).sub(held);
  const bubbles = (frequency: number) =>
    mix(bubblesAt(lace, drift.b, frequency, scale), bubblesAt(lace, drift.a, frequency, scale), drift.blend);
  // 1 outside a round hole of the given radius around each bubble's centre
  const outside = (distance: Float, radius: Float) => {
    const width = fwidth(distance).add(0.015);
    return smoothstep(radius.sub(width), radius.add(width), distance);
  };
  const holes = (b: Node<'vec2'>, smallest: number, largest: number) =>
    outside(b.x, mix(smallest, largest, b.y).mul(thin.mul(0.9).add(0.45)));
  const small = bubbles(2.7);
  const foam = holes(bubbles(1), 0.12, 0.5).mul(holes(small, 0.08, 0.42));
  // a few large bubbles open wide, more as the foam thins; their rims are ragged with the small bubbles
  const large = bubbles(0.38);
  const opening = smoothstep(0.6, 0.95, large.y).mul(thin.mul(0.6).add(0.4)).mul(0.7);
  const open = float(1)
    .sub(outside(large.x.add(small.x.mul(0.12)), opening))
    .mul(smoothstep(0, 0.05, opening));
  const coverage = foam.mul(float(1).sub(open)).mul(smoothstep(0.01, 0.12, held));
  // Applied after the foam strength, so it can't push churned foam back to solid white: thin foam is a
  // translucent film, and even the thickest lets a little water through, more under some bubbles than others.
  const film = mix(0.4, 0.85, held).mul(mix(0.85, 1, small.y));
  return clamp(coverage.mul(u.uOceanFoam), 0, 1).mul(film);
}

/** Two octaves of value noise, each fading to its mean once its cells get smaller than a pixel. */
function foamNoise(obj: Vec3, u: DeformUniforms): Float {
  // Narrow in the swell direction and stretched along each crest.
  const scale = float(FOAM_REFERENCE_WAVELENGTH).div(u.uOceanWavelength);
  const q = vec3(
    dot(obj, vec3(0.35, 0.9, 0.25))
      .mul(79)
      .mul(scale)
      .sub(u.uFoamScroll.mul(1.3)),
    dot(obj, vec3(-0.88, 0.32, 0.34))
      .mul(17)
      .mul(scale)
      .add(u.uFoamScroll.mul(0.18)),
    dot(obj, vec3(0.3, -0.16, 0.94))
      .mul(12)
      .mul(scale),
  );
  const cells = length(fwidth(q));
  const octave = (frequency: number) =>
    mix(vnoise(q.mul(frequency)), float(0.5), smoothstep(0.35, 0.8, cells.mul(frequency)));
  return octave(1).mul(0.7).add(octave(2.35).mul(0.3));
}

/** Whitecaps follow steep travelling crests; simulated foam persists and drifts after impacts. */
export function oceanFoam(
  drift: FoamDrift,
  lace: Texture,
  breaking: Float,
  weather: Float,
  simulated: Float,
  u: DeformUniforms,
  shore?: Float,
): Float {
  // the two drifted patterns crossfaded, keeping the noise's contrast through the blend
  const wa = drift.blend;
  const wb = float(1).sub(wa);
  const n = foamNoise(drift.a, u)
    .sub(0.5)
    .mul(wa)
    .add(foamNoise(drift.b, u).sub(0.5).mul(wb))
    .div(sqrt(wa.mul(wa).add(wb.mul(wb))))
    .add(0.5);
  // Whitecaps where the waves fold over, broken into streaks; the noise varies how readily each spot foams.
  const crests = smoothstep(mix(0.3, 0.22, weather), 0.75, breaking.mul(n.mul(0.9).add(0.55)));
  // churned foam left by the sim: taps, drags, splashes and old whitecaps
  const sprayFoam = foamLace(drift, lace, smoothstep(0.04, 0.6, simulated), u);
  let foam = crests;
  if (shore) {
    const wash = smoothstep(0.1, 0.62, simulated)
      .mul(float(1).sub(smoothstep(0.03, 0.13, shore)))
      .mul(mix(0.5, 0.9, n));
    foam = max(foam, wash);
  }
  return max(clamp(foam.mul(u.uOceanFoam), 0, 1), sprayFoam);
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

/**
 * The sun on the water, added as emission. A GGX glint whose width follows the surface roughness, so waves too
 * small to see spread it into a glitter path; light scattered back out of thin crests seen against the sun
 * (after Atlas, GDC 2019); and sunlit foam. `normal` is the view-space shading normal, `height` the wave height.
 */
export function oceanSunLight(
  normal: Vec3,
  height: Float,
  foam: Float,
  roughness: Float,
  u: DeformUniforms,
): Vec3 {
  const n = normalize(normal);
  const v = positionViewDirection;
  const l = normalize(cameraViewMatrix.mul(vec4(u.uSunDir, 0)).xyz);
  const h = normalize(l.add(v));
  const nl = clamp(dot(n, l), 0, 1);
  const nv = clamp(dot(n, v), 1e-3, 1);
  const nh = clamp(dot(n, h), 0, 1);
  // the sun's disc keeps even glassy water's glint a little wide
  const alpha = roughness.mul(roughness);
  const a2 = max(alpha.mul(alpha), 0.0004);
  const lobe = nh.mul(nh).mul(a2.sub(1)).add(1);
  const distribution = a2.div(lobe.mul(lobe).mul(Math.PI));
  const visibility = float(0.5).div(
    nl
      .mul(sqrt(nv.mul(nv).mul(float(1).sub(a2)).add(a2)))
      .add(nv.mul(sqrt(nl.mul(nl).mul(float(1).sub(a2)).add(a2))))
      .add(1e-5),
  );
  const fresnel = pow(float(1).sub(clamp(dot(v, h), 0, 1)), 5)
    .mul(0.98)
    .add(0.02);
  const glint = distribution.mul(visibility).mul(fresnel).mul(nl);

  const crest = smoothstep(0.002, 0.012, height);
  const towardSun = pow(clamp(dot(v, l.negate()), 0, 1), 4);
  const litFromBehind = pow(clamp(dot(l, n).mul(-0.5).add(0.5), 0, 1), 3);
  const scatter = crest
    .mul(towardSun)
    .mul(litFromBehind)
    .mul(0.8)
    .add(crest.mul(nv.mul(nv)).mul(0.03))
    .add(nl.mul(0.015));
  const water = vec3(0.07, 0.36, 0.33).mul(scatter).add(vec3(glint));
  return mix(water, vec3(nl.mul(0.25)), foam).mul(u.uSunColor);
}
