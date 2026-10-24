import type { Texture } from 'three';
import type { Node } from 'three/webgpu';
import {
  abs,
  atan,
  clamp,
  dot,
  float,
  length,
  log2,
  max,
  min,
  select,
  smoothstep,
  sqrt,
  texture,
  vec2,
  vec3,
} from 'three/tsl';
import type { OceanCascade } from '../core/oceanFFT';
import { OCEAN_GUSTS } from '../core/oceanWind';
import type { DeformUniforms } from './deform';

// Samples the FFT ocean (materials/oceanFFT.ts) on the orb. Directions map to cube faces in equal-angle
// coordinates, so wavelengths stay within a few percent across a face. Every cascade tiles a face a whole
// number of times, so the four faces around the equator join seamlessly; only the two polar faces are
// blended in, with weights that keep the waves' variance (two unrelated seas averaged would look calmer).

type Vec3 = Node<'vec3'>;
type Float = Node<'float'>;

export interface OceanWaveFields {
  readonly size: number;
  readonly cascades: readonly OceanCascade[];
  readonly displacement: readonly Texture[];
  readonly derivatives: readonly Texture[];
  readonly foam: readonly Texture[];
  readonly slopeVariance: readonly number[];
}

/** Angular radius of a cat's paw. */
const GUST_RADIUS = 0.3;

/** How hard the gusting wind is roughening the water at p: 0 in calm air, about 1 under a cat's paw. */
export function oceanGustAt(p: Vec3, u: DeformUniforms): Float {
  let gust: Float = float(0);
  for (let i = 0; i < OCEAN_GUSTS; i++) {
    const patch = u.uGusts.element(i);
    gust = gust.add(
      smoothstep(Math.cos(GUST_RADIUS), Math.cos(GUST_RADIUS * 0.3), dot(p, patch.xyz)).mul(patch.w),
    );
  }
  return min(gust, 1);
}

/** One cube face: its axis n, and the axes e and up its u and v coordinates run along. */
interface Face {
  n: Vec3;
  e: Vec3;
  up: Vec3;
}

/** Wave sums over the cascades for one face, in that face's u and v axes. */
interface FaceWaves {
  height: Float;
  du: Float;
  dv: Float;
  su: Float;
  sv: Float;
  duu: Float;
  dvv: Float;
  duv: Float;
  lost: Float;
  foam: Float;
}

const QUARTER = Math.PI / 2;
/** How much a cat's paw steepens the short waves' slopes at its heart. */
const GUST_ROUGHENING = 1.5;
const sideOf = (x: Float) => select(x.greaterThanEqual(0), float(1), float(-1));

/** The equator face facing p: x or z. East runs around the equator, so it lines up across the four faces. */
function ringFace(p: Vec3): Face {
  const useX = abs(p.x).greaterThan(abs(p.z));
  const n = select(useX, vec3(sideOf(p.x), 0, 0), vec3(0, 0, sideOf(p.z)));
  // east = up x n
  const e = select(useX, vec3(0, 0, sideOf(p.x).negate()), vec3(sideOf(p.z), 0, 0));
  return { n, e, up: vec3(0, 1, 0) };
}

function polarFace(p: Vec3): Face {
  return { n: vec3(0, sideOf(p.y), 0), e: vec3(1, 0, 0), up: vec3(0, 0, 1) };
}

/** Equal-angle face coordinates of p, their surface gradients, and how far p moves per unit of each. */
function faceCoords(p: Vec3, face: Face) {
  const dn = max(dot(p, face.n), 1e-4);
  const pe = dot(p, face.e);
  const pu = dot(p, face.up);
  const e2 = max(dn.mul(dn).add(pe.mul(pe)), 1e-8);
  const u2 = max(dn.mul(dn).add(pu.mul(pu)), 1e-8);
  return {
    u: atan(pe, dn),
    v: atan(pu, dn),
    // d(atan(pe / dn)) / dp; it is already tangent to the sphere
    gradU: face.e.mul(dn).sub(face.n.mul(pe)).div(e2),
    gradV: face.up.mul(dn).sub(face.n.mul(pu)).div(u2),
    alongU: face.e.sub(p.mul(pe)).mul(e2.div(dn)),
    alongV: face.up.sub(p.mul(pu)).mul(u2.div(dn)),
  };
}

/**
 * Sums the cascades on one face. `weight(c)` is that face's blend weight for cascade c and `lod(c)` the mip
 * level to read; the tile's turn is undone so every result is in the face's u and v axes.
 */
function sampleFace(
  p: Vec3,
  face: Face,
  fields: OceanWaveFields,
  u: DeformUniforms,
  weather: Float,
  gust: Float,
  weight: (c: number) => Float,
  lod: (c: number) => Float,
  withFoam = false,
): FaceWaves & ReturnType<typeof faceCoords> {
  const coords = faceCoords(p, face);
  const sum: FaceWaves = {
    height: float(0),
    du: float(0),
    dv: float(0),
    su: float(0),
    sv: float(0),
    duu: float(0),
    dvv: float(0),
    duv: float(0),
    lost: float(0),
    foam: float(0),
  };
  const chop = u.uOceanChoppiness.mul(0.5);
  fields.cascades.forEach((cascade, c) => {
    const { a, b, cos: co, sin: si } = cascade;
    const uv = vec2(coords.u.mul(a).sub(coords.v.mul(b)), coords.u.mul(b).add(coords.v.mul(a))).div(QUARTER);
    const level = lod(c);
    const disp = texture(fields.displacement[c]!, uv).level(level);
    const deriv = texture(fields.derivatives[c]!, uv).level(level);
    // storms build the short waves more than the long swell
    const w = weight(c)
      .mul(u.uOceanAmplitude)
      .mul(weather.mul(c === 0 ? 0.2 : 0.75).add(1));
    const wChop = w.mul(chop);
    // a gust only ripples the short waves' surface: steeper slopes and more roughness, no height or breaking
    const wSlope = c === 0 ? w : w.mul(gust.mul(GUST_ROUGHENING).add(1));
    // tile axes back to face axes: R(-turn)
    sum.height = sum.height.add(disp.y.mul(w));
    sum.du = sum.du.add(disp.x.mul(co).add(disp.z.mul(si)).mul(wChop));
    sum.dv = sum.dv.add(disp.z.mul(co).sub(disp.x.mul(si)).mul(wChop));
    sum.su = sum.su.add(deriv.x.mul(co).add(deriv.y.mul(si)).mul(wSlope));
    sum.sv = sum.sv.add(deriv.y.mul(co).sub(deriv.x.mul(si)).mul(wSlope));
    // the displacement gradient turns as a tensor: R^T T R
    const [xx, zz, xz] = [deriv.z, deriv.w, disp.w];
    const cc = co * co;
    const ss = si * si;
    const cs = co * si;
    sum.duu = sum.duu.add(
      xx
        .mul(cc)
        .add(xz.mul(2 * cs))
        .add(zz.mul(ss))
        .mul(wChop),
    );
    sum.dvv = sum.dvv.add(
      xx
        .mul(ss)
        .sub(xz.mul(2 * cs))
        .add(zz.mul(cc))
        .mul(wChop),
    );
    sum.duv = sum.duv.add(
      xx
        .mul(-cs)
        .add(xz.mul(cc - ss))
        .add(zz.mul(cs))
        .mul(wChop),
    );
    // Slope variance the mip averaged away; it comes back as roughness. Near the base level the local slope
    // is just noisy, so only count it once the texture is really filtered.
    const resolved = deriv.x.mul(deriv.x).add(deriv.y.mul(deriv.y));
    const lost = max(float(fields.slopeVariance[c]!).sub(resolved), 0).mul(smoothstep(0.5, 2.5, level));
    sum.lost = sum.lost.add(lost.mul(wSlope).mul(wSlope));
    if (withFoam) {
      // folding was measured at the tuned amplitude; storms deepen it
      const memory = texture(fields.foam[c]!, uv).level(level).x;
      sum.foam = sum.foam.add(memory.mul(weight(c)).mul(weather.mul(c === 0 ? 0.2 : 0.75).add(1)));
    }
  });
  return { ...sum, ...coords };
}

/** Variance-keeping blend weights into the polar face for each cascade: [ring, polar]. */
function blend(p: Vec3, fields: OceanWaveFields): (c: number) => [Float, Float] {
  const q = abs(p.y).div(max(max(abs(p.x), abs(p.z)), 1e-4));
  return (c) => {
    const band = fields.cascades[c]!.band;
    const polar = smoothstep(1 - band, 1 + band, q);
    const ring = float(1).sub(polar);
    const norm = sqrt(ring.mul(ring).add(polar.mul(polar)));
    return [ring.div(norm), polar.div(norm)];
  };
}

/**
 * Vertex-stage waves at the undisplaced unit direction p: the tangent offset (on the unit sphere), height and
 * height gradient. The mips are read at the mesh's own resolution, so waves the mesh can't hold don't alias;
 * they come back per pixel in oceanWaveSurface.
 */
export function oceanWaveVertex(
  p: Vec3,
  fields: OceanWaveFields,
  u: DeformUniforms,
  weather: Float,
  spacing: number,
): { offset: Vec3; height: Float; gradient: Vec3 } {
  const weights = blend(p, fields);
  const lod = (c: number) =>
    float(Math.max(0, Math.log2((spacing * fields.size) / fields.cascades[c]!.tile)));
  const gust = oceanGustAt(p, u);
  const ring = sampleFace(p, ringFace(p), fields, u, weather, gust, (c) => weights(c)[0], lod);
  const polar = sampleFace(p, polarFace(p), fields, u, weather, gust, (c) => weights(c)[1], lod);
  const offset = ring.alongU
    .mul(ring.du)
    .add(ring.alongV.mul(ring.dv))
    .add(polar.alongU.mul(polar.du))
    .add(polar.alongV.mul(polar.dv));
  const gradient = ring.gradU
    .mul(ring.su)
    .add(ring.gradV.mul(ring.sv))
    .add(polar.gradU.mul(polar.su))
    .add(polar.gradV.mul(polar.sv));
  return { offset, height: ring.height.add(polar.height), gradient };
}

/**
 * The remembered folding whitecaps grow from, at undisplaced direction p, averaged over the mesh's vertex
 * spacing: how much of the water around a vertex has broken in the last few seconds. Compute passes can read
 * it (the mips are chosen explicitly).
 */
export function oceanWaveFoam(
  p: Vec3,
  fields: OceanWaveFields,
  u: DeformUniforms,
  weather: Float,
  spacing: number,
): Float {
  const weights = blend(p, fields);
  const lod = (c: number) =>
    float(Math.max(0, Math.log2((spacing * fields.size) / fields.cascades[c]!.tile)));
  // gusts roughen the water without breaking it
  const ring = sampleFace(p, ringFace(p), fields, u, weather, float(0), (c) => weights(c)[0], lod, true);
  const polar = sampleFace(p, polarFace(p), fields, u, weather, float(0), (c) => weights(c)[1], lod, true);
  return ring.foam.add(polar.foam);
}

/**
 * Fragment-stage waves at the interpolated undisplaced direction p: the height gradient on the unit sphere
 * (corrected for how chop squeezes crests), the Jacobian of the horizontal displacement (below 1 where the
 * surface bunches up, below 0 where crests fold over and break), the slope variance the mips filtered out, and
 * the remembered folding that whitecaps grow from.
 */
export function oceanWaveSurface(
  p: Vec3,
  fields: OceanWaveFields,
  u: DeformUniforms,
  weather: Float,
): { gradient: Vec3; jacobian: Float; lostVariance: Float; foam: Float } {
  const weights = blend(p, fields);
  // one texel per pixel footprint
  const footprint = max(length(p.dFdx()), length(p.dFdy()));
  const lod = (c: number) => max(log2(footprint.mul(fields.size / fields.cascades[c]!.tile)), 0);
  const gust = oceanGustAt(p, u);
  const ring = sampleFace(p, ringFace(p), fields, u, weather, gust, (c) => weights(c)[0], lod, true);
  const polar = sampleFace(p, polarFace(p), fields, u, weather, gust, (c) => weights(c)[1], lod, true);
  const corrected = (f: typeof ring) =>
    f.gradU.mul(f.su.div(max(f.duu.add(1), 0.25))).add(f.gradV.mul(f.sv.div(max(f.dvv.add(1), 0.25))));
  const jacobian = (f: typeof ring) => f.duu.add(1).mul(f.dvv.add(1)).sub(f.duv.mul(f.duv)).sub(1);
  return {
    gradient: corrected(ring).add(corrected(polar)),
    jacobian: clamp(jacobian(ring).add(jacobian(polar)).add(1), -1, 2),
    lostVariance: ring.lost.add(polar.lost),
    foam: ring.foam.add(polar.foam),
  };
}
