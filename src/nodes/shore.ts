import type { Node } from 'three/webgpu';
import {
  abs,
  attribute,
  cameraViewMatrix,
  color,
  dot,
  float,
  length,
  max,
  mix,
  normalize,
  positionGeometry,
  smoothstep,
  vec4,
} from 'three/tsl';
import { bumped, sandGrains, sandRelief, type SandLight, type SandSlope } from './sand';
import { baseR } from './shape';

type Float = Node<'float'>;
type Vec3 = Node<'vec3'>;

// Close-up detail for the coast's sand (materials/oceanShores.ts), which its vertices are far too coarse to carry:
// grains, wind ripples and a bump, darker and glossy where the sim's water has wet it. The water running up the
// beach is the sim's own, drawn by the sea's surface.

/** The beach's wet sand (WET_SAND in materials/oceanShores.ts). */
const WET_SAND = color('#7c5a45');
/** How much of the key light's diffuse the land's emissive carries (1 / pi, for a Lambert surface). */
const KEY_SHARE = 1 / Math.PI;

/**
 * The shore's sand: `water` is how much of the sim's own water stands here (0 to 1), and `soakedBy` how wet the sim
 * has left the sand (0 to 1, drying over seconds), `dug` how far the hand has dug (negative) or piled the sand
 * here, `slope` where the digging left it steep; dug sand is damp and darker, like the sand orb's. Its grains glint
 * in `light`, which also rakes across the land as its key light (as emissive, so it doesn't light the sea).
 */
export function shoreSurface(
  water: Float,
  soakedBy: Float,
  light: SandLight,
  dug: Float = float(0),
  slope?: SandSlope,
): { color: Vec3; roughness: Float; normal: Vec3; emissive: Vec3 } {
  const wet = attribute('terrain', 'float');
  const p = normalize(positionGeometry);

  // sand: ripples the wind combed across the dry sand, wiped where it's dug; crumbs thrown out of a dig
  const disturbed = smoothstep(0.0005, 0.004, abs(dug));
  const relief = sandRelief(p, float(1).sub(wet).mul(float(1).sub(disturbed)), disturbed, slope);
  const tone = relief.tone;
  const height = relief.height.mul(float(1).sub(wet.mul(0.7)));
  // height above the still sea, as the land mesh was built (it follows the cube morph like the water)
  const above = length(positionGeometry).sub(baseR(p));
  // seabed the sea has drawn back from is bare wet sand, not the pale sand seen through clear water
  const bared = float(1)
    .sub(smoothstep(-0.0005, 0.0005, above))
    .mul(float(1).sub(water));
  // the sand is as wet as the sim's water has left it: dark where a wave ran up, drying back over seconds; sand
  // turned up from underneath is damp too
  const soaked = max(
    // (the sand's own damp band just above the still sea is faint: the sim's water does the rest)
    max(wet.mul(0.2), soakedBy.mul(0.95)),
    max(smoothstep(0.0005, 0.005, abs(dug)).mul(0.8), relief.damp),
  );
  const grains = sandGrains(p, bumped(height), light, soaked);
  // wet sand is darker and glossy (bared seabed takes the beach's own wet sand, not the paler sand meant to be
  // seen through water)
  const albedo = mix(attribute('color', 'vec3'), WET_SAND, bared)
    .mul(tone)
    .mul(grains.tint)
    .mul(mix(float(1), float(0.62), soaked));
  const l = normalize(cameraViewMatrix.mul(vec4(light.dir, 0)).xyz);
  const key = albedo
    .mul(light.color)
    .mul(max(dot(grains.normal, l), 0))
    .mul(KEY_SHARE);
  return {
    color: albedo,
    roughness: mix(float(0.88), float(0.35), soaked),
    normal: grains.normal,
    emissive: key.add(grains.glint),
  };
}
