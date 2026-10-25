import type { Node } from 'three/webgpu';
import {
  abs,
  attribute,
  cameraViewMatrix,
  pow,
  color,
  cos,
  dot,
  float,
  fract,
  length,
  max,
  mix,
  normalize,
  positionGeometry,
  select,
  sin,
  smoothstep,
  vec3,
  vec4,
} from 'three/tsl';
import { shelfLag } from '../core/oceanShores';
import { snoise } from './noise';
import { threadsAt } from './ocean';
import { bumped, sandGrains, sandRelief, type SandLight, type SandSlope } from './sand';
import { baseR } from './shape';
import { surfWave, type SurfInputs } from './surf';

type Float = Node<'float'>;
type Vec3 = Node<'vec3'>;

// Close-up detail for the coast's sand (materials/oceanShores.ts), which its vertices are far too coarse to carry:
// grains, wind ripples and a bump, darker and glossy where the per-vertex `terrain` attribute says the sand is wet,
// with the swash running over it.

/** Height the swash runs up a shore facing the swell, per unit of the sea's state, orb radii. */
const SWASH_RUNUP = 0.01;
/** The beach's wet sand (WET_SAND in materials/oceanShores.ts). */
const WET_SAND = color('#7c5a45');
/** How much of the key light's diffuse the land's emissive carries (1 / pi, for a Lambert surface). */
const KEY_SHARE = 1 / Math.PI;
/** Share of a wave's cycle the swash spends running up; the rest it drains back. */
const UPRUSH = 0.3;

/**
 * Swash: after each wave breaks at the shoreline (surfWave, as the sim's surf), a sheet of water runs up the beach
 * and drains back. Its reach is a height, so a gentle beach is covered far inland and a steep one barely wet.
 * Returns how much of a point the sheet covers, its depth there, the foam on it (a bubbly line at its front,
 * lace left behind as it drains), and how wet the run-up has left the sand.
 */
function swashAt(p: Vec3, height: Float, surf: SurfInputs) {
  const swash = attribute('swash', 'vec4');
  const foot = normalize(p.add(swash.xyz));
  const seaward = normalize(swash.xyz.add(p.cross(vec3(0, 0, 1)).mul(1e-5)));
  const wave = surfWave(foot, float(0), float(shelfLag(0)), surf);
  // Phase falls with time, so the cycle advances as it moves on from a crest at the shoreline.
  const cycle = fract(
    float(Math.PI / 2)
      .sub(wave.phases[0]!)
      .div(Math.PI * 2),
  );
  const reach = select(
    cycle.lessThan(UPRUSH),
    sin(cycle.div(UPRUSH).mul(Math.PI / 2)),
    cos(
      cycle
        .sub(UPRUSH)
        .div(1 - UPRUSH)
        .mul(Math.PI / 2),
    ).pow(1.5),
  );
  // each stretch of beach runs up its own distance: the front is ragged, not a contour line
  const ragged = snoise(foot.mul(45).add(swash.w))
    .mul(0.12)
    .add(snoise(foot.mul(130)).mul(0.05))
    .add(1);
  const runup = float(SWASH_RUNUP).mul(surf.energy).mul(wave.set).mul(ragged);
  // (a wave the hand sends is the sim's own water, which floods the beach itself)
  const level = runup.mul(reach);
  const depth = max(level.sub(height), 0);
  const cover = float(1)
    .sub(smoothstep(level.sub(0.0004), level, height))
    .mul(smoothstep(0.0002, 0.0006, level));
  // The sea's own foam carried up the beach: the same threads and torn white water (threadsAt, nodes/ocean.ts),
  // drifting with the water, up on the uprush and back down on the backwash, so the run-up reads as the sea's
  // surf running on rather than a separate line. Thickest at the front, thinning to lace behind it as it drains.
  const drift = p.add(seaward.mul(level.mul(-6)));
  const band = max(level.mul(0.3), 0.002);
  const front = float(1)
    .sub(smoothstep(0, band, level.sub(height)))
    .mul(smoothstep(0.0002, 0.0008, level))
    .mul(cover);
  const draining = smoothstep(UPRUSH, UPRUSH + 0.25, cycle);
  const amount = cover.mul(mix(float(0.45), float(0.22), draining)).add(front.mul(0.5));
  // (finer than the sea's and less drawn out: a thin sheet's foam is fine lace, not long streaks)
  const foam = threadsAt(
    drift.mul(1.7),
    seaward.mul(0.55),
    amount,
    front.mul(mix(float(0.7), float(0.35), draining)),
  );
  // sand the run-up has reached stays dark and glossy for a while
  const soaked = float(1)
    .sub(smoothstep(runup.mul(0.7), runup.add(0.0006), height))
    .mul(0.8);
  return { cover, depth, foam, soaked };
}

/**
 * The shore's sand: `water` is how much of the sim's own water stands here (0 to 1), `flooded` how much of it is
 * standing on the land itself rather than the sea at the shore, and `soakedBy` how wet the sim
 * has left the sand (0 to 1, drying over seconds), `dug` how far the hand has dug (negative) or piled the sand
 * here, `slope` where the digging left it steep; dug sand is damp and darker, like the sand orb's. Its grains glint
 * in `light`, which also rakes across the land as its key light (as emissive, so it doesn't light the sea).
 */
export function shoreSurface(
  surf: SurfInputs,
  { water, flooded }: { water: Float; flooded: Float },
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
  // (the beach as built: water in holes the hand digs is the sim's, drawn by the sea's own surface)
  const above = length(positionGeometry).sub(baseR(p));
  // the swash runs only over the beach: below the still sea the seabed is the sea's, which draws its own water,
  // and so is the beach wherever the sim's water stands on it (a wave the hand sent)
  const ashore = smoothstep(-0.0005, 0.0005, above);
  const open = ashore.mul(float(1).sub(flooded));
  const swash = swashAt(p, above, surf);
  const sw = { ...swash, cover: swash.cover.mul(open), foam: swash.foam.mul(open) };
  // sand turned up from underneath is damp
  // seabed the sea has drawn back from is bare wet sand, not the pale sand seen through clear water
  const bared = float(1).sub(ashore).mul(float(1).sub(water));
  const soaked = max(
    max(max(wet, sw.soaked), soakedBy.mul(0.9)),
    max(smoothstep(0.0005, 0.005, abs(dug)).mul(0.8), relief.damp),
  );
  const grains = sandGrains(
    p,
    bumped(height.mul(float(1).sub(sw.cover.mul(0.8)))),
    light,
    max(soaked, sw.cover),
  );
  // wet sand is darker and glossy; under the swash, clear water over it, and white foam on top
  // (bared seabed takes the beach's own wet sand, not the paler sand meant to be seen through water)
  const ground = mix(attribute('color', 'vec3'), WET_SAND, bared)
    .mul(tone)
    .mul(grains.tint)
    .mul(mix(float(1), float(0.62), soaked));
  // under the run-up's thin sheet the sand is darker still, and barely tinted: it's millimetres of clear water
  // (absorbed red first, as the sea is: at the waterline it's as tinted as the sea's own edge, so they join up)
  const sheet = ground.mul(pow(vec3(0.58, 0.76, 0.86), vec3(sw.depth.div(0.01))));
  const covered = mix(ground, sheet, sw.cover);
  const albedo = mix(covered, vec3(0.92, 0.96, 0.97), sw.foam);
  const l = normalize(cameraViewMatrix.mul(vec4(light.dir, 0)).xyz);
  const key = albedo
    .mul(light.color)
    .mul(max(dot(grains.normal, l), 0))
    .mul(KEY_SHARE);
  return {
    color: albedo,
    roughness: mix(mix(float(0.88), float(0.35), soaked), float(0.2), sw.cover.mul(float(1).sub(sw.foam))),
    normal: grains.normal,
    emissive: key.add(grains.glint),
  };
}
