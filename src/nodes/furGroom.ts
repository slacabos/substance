import type { Node } from 'three/webgpu';
import { dot, normalize, sin, vec3 } from 'three/tsl';
import { FUR } from '../core/fur';

type Vec3 = Node<'vec3'>;
type Float = Node<'float'>;

/** Object-space coat lay, matching furGroom in core/fur.ts. No normalized tangent, so cowlicks stay finite. */
export function furGroom(p: Vec3, n: Vec3): Vec3 {
  const { direction, xWave, zWave } = FUR.groom;
  const flow = vec3(
    sin(p.y.mul(xWave.y).add(p.z.mul(xWave.z)))
      .mul(xWave.amplitude)
      .add(direction[0]),
    direction[1],
    sin(p.x.mul(zWave.x)).mul(zWave.amplitude).add(direction[2]),
  );
  return flow.sub(n.mul(dot(flow, n)));
}

/** Mean direction of a gently curving fibre; roots stand up and the tips follow the coat's lay. */
export function furRestOffset(n: Vec3, groom: Vec3, t: Float, distance: Float): Vec3 {
  return normalize(n.add(groom.mul(t.mul(FUR.groom.bend * 0.5).add(FUR.groom.rootLean)))).mul(distance);
}
