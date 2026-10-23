import type { Color } from 'three';
import type { Node, UniformNode } from 'three/webgpu';
import { atan, exp, length, mix, normalize, positionLocal, pow, smoothstep, uv, vec4 } from 'three/tsl';

type ColorUniform = UniformNode<'color', Color>;
type FloatUniform = UniformNode<'float', number>;

/** Studio sweep: a wall-to-floor gradient, a band where they meet and a soft light behind the object. */
export function backdropColor(u: {
  wall: ColorUniform;
  floor: ColorUniform;
  horizon: ColorUniform;
  halo: ColorUniform;
}): Node<'vec3'> {
  const d = normalize(positionLocal);
  const az = atan(d.x, d.z.negate());
  return mix(u.floor, u.wall, smoothstep(-0.5, 0.8, d.y))
    .add(u.horizon.mul(exp(pow(d.y.add(0.08), 2).mul(-25))))
    .add(
      u.halo.mul(
        exp(
          pow(az.mul(1.3), 2)
            .negate()
            .sub(pow(d.y.sub(0.8).mul(3), 2)),
        ),
      ),
    );
}

/**
 * Soft pool under the object: an additive light glow on dark sets, a contact shadow on bright ones.
 * Meant for premultiplied blending (One, OneMinusSrcAlpha), so rgb adds light and alpha takes it away.
 */
export function poolColor(u: {
  color: ColorUniform;
  strength: FloatUniform;
  glow: FloatUniform;
  shadow: FloatUniform;
}): Node<'vec4'> {
  const d = length(uv().sub(0.5)).mul(2);
  const falloff = pow(smoothstep(0, 1, d).oneMinus(), 3);
  const shadow = u.shadow.mul(pow(smoothstep(0, 0.7, d).oneMinus(), 1.6));
  return vec4(u.color.mul(u.strength).mul(u.glow).mul(falloff), shadow);
}
