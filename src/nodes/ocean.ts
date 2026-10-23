import type { Node } from 'three/webgpu';
import { clamp, floor, fract, mix, smoothstep, vec3 } from 'three/tsl';
import type { DeformUniforms } from './deform';

// The ocean's surface: water colour by wave height, lacy foam on the crests.

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

/** How much foam covers this point: crests above a threshold, broken up into lace. */
export function oceanFoam(obj: Vec3, wave: Float, u: DeformUniforms): Float {
  const q = obj.mul(38).add(vec3(0, u.uTime.mul(0.35), 0));
  const n = vnoise(q)
    .mul(0.6)
    .add(vnoise(q.mul(2.7)).mul(0.4));
  const lace = smoothstep(0.42, 0.62, n);
  return clamp(smoothstep(0.02, 0.045, wave.add(n.sub(0.5).mul(0.018))).mul(mix(0.55, 1, lace)), 0, 1);
}

/** Deep water in the troughs, lighter on the swells, white where there's foam. */
export function oceanColor(wave: Float, foam: Float, u: DeformUniforms): Vec3 {
  const water = mix(u.uDeep, u.uShallow, smoothstep(-0.03, 0.045, wave));
  return mix(water, vec3(0.92, 0.96, 0.98), foam);
}
