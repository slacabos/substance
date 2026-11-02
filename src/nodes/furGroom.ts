import type { Node } from 'three/webgpu';
import { cos, dot, float, length, max, mix, normalize, select, sin, vec3 } from 'three/tsl';
import { FUR } from '../core/fur';
import { planeUniform } from '../scene/shape';
import { planePointNode, planePullNode } from './chart';

type Vec3 = Node<'vec3'>;
type Float = Node<'float'>;

/**
 * Object-space coat lay, matching furGroom in core/fur.ts on the orb. No normalized tangent, so cowlicks stay finite.
 * On the plane's ground it's combed one way across the ground instead, swaying as it goes (FUR.plane).
 */
export function furGroom(p: Vec3, n: Vec3): Vec3 {
  const { direction, xWave, zWave } = FUR.groom;
  const flow = vec3(
    sin(p.y.mul(xWave.y).add(p.z.mul(xWave.z)))
      .mul(xWave.amplitude)
      .add(direction[0]),
    direction[1],
    sin(p.x.mul(zWave.x)).mul(zWave.amplitude).add(direction[2]),
  );
  const orb = flow.sub(n.mul(dot(flow, n)));
  const { lay, wave } = FUR.plane;
  const ground = planePointNode(p);
  const k = (2 * Math.PI) / wave.length;
  const sway = sin(ground.x.mul(k * 0.8).add(ground.z.mul(k * 0.6)))
    .add(sin(ground.z.mul(k * 1.3).sub(ground.x.mul(k * 0.4))).mul(0.5))
    .mul(wave.angle / 1.5);
  const c = cos(sway);
  const s = sin(sway);
  const combed = vec3(c.mul(lay[0]).sub(s.mul(lay[1])), 0, s.mul(lay[0]).add(c.mul(lay[1])));
  const pulled = planePullNode(p, combed);
  const plane = pulled.mul(float(Math.hypot(...direction) * FUR.plane.lean).div(max(length(pulled), 1e-20)));
  // (the orb's lay exactly, off the plane: nothing from the chart can reach its sim there)
  return select(planeUniform.greaterThan(0), mix(orb, plane, planeUniform), orb);
}

/** Mean direction of a gently curving fibre; roots stand up and the tips follow the coat's lay. */
export function furRestOffset(n: Vec3, groom: Vec3, t: Float, distance: Float): Vec3 {
  return normalize(n.add(groom.mul(t.mul(FUR.groom.bend * 0.5).add(FUR.groom.rootLean)))).mul(distance);
}
