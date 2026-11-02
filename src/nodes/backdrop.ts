import type { Color } from 'three';
import type { Node, UniformNode } from 'three/webgpu';
import {
  atan,
  cameraPosition,
  dot,
  exp,
  float,
  fog,
  length,
  max,
  mix,
  normalize,
  positionLocal,
  positionWorld,
  pow,
  smoothstep,
  sqrt,
  uv,
  vec4,
} from 'three/tsl';
import { planeUniform } from '../scene/shape';

type ColorUniform = UniformNode<'color', Color>;
type FloatUniform = UniformNode<'float', number>;

export interface BackdropUniforms {
  wall: ColorUniform;
  floor: ColorUniform;
  horizon: ColorUniform;
  halo: ColorUniform;
  hazeHalf: FloatUniform;
}

/**
 * Studio sweep: a wall-to-floor gradient, a band where they meet and a soft light behind the object. `d` is the
 * direction from the backdrop's centre, the sphere's own by default.
 */
export function backdropColor(u: BackdropUniforms, d: Node<'vec3'> = normalize(positionLocal)): Node<'vec3'> {
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

/** The backdrop sphere's radius (scene/backdrop.ts). */
export const BACKDROP_RADIUS = 40;

/** Fully conceal the chart's far boundary; half-density distance comes from the setting. */
const HAZE_FULL: readonly [number, number] = [90, 140];

/**
 * Haze over the plane, the colour of the sky right behind each point, so the ground fades into the backdrop
 * without a seam: the view ray carried on to the backdrop sphere gives the direction its colour is read at.
 * Nothing on the orb or the cube is hazed.
 */
export function hazeAt(u: BackdropUniforms, point: Node<'vec3'>, eye: Node<'vec3'>) {
  const ray = point.sub(eye);
  const far = length(ray);
  const v = ray.div(max(far, 1e-6));
  const b = dot(eye, v);
  const reach = b.negate().add(
    sqrt(
      max(
        b
          .mul(b)
          .sub(dot(eye, eye))
          .add(BACKDROP_RADIUS ** 2),
        0,
      ),
    ),
  );
  const sky = backdropColor(u, normalize(eye.add(v.mul(reach))));
  const density = float(Math.sqrt(Math.log(2))).div(u.hazeHalf);
  const thick = float(1).sub(exp(far.mul(density).mul(far.mul(density)).negate()));
  return { color: sky, amount: max(thick, smoothstep(HAZE_FULL[0], HAZE_FULL[1], far)).mul(planeUniform) };
}

/** Visible scene colour only; data passes opt out of fog. */
export function backdropHaze(u: BackdropUniforms): Node<'vec4'> {
  const haze = hazeAt(u, positionWorld, cameraPosition);
  return fog(haze.color, haze.amount) as Node<'vec4'>;
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
  /** Fades the whole pool out, as the plane takes the floor away. */
  shown: FloatUniform;
}): Node<'vec4'> {
  const d = length(uv().sub(0.5)).mul(2);
  const falloff = pow(smoothstep(0, 1, d).oneMinus(), 3);
  const shadow = u.shadow.mul(pow(smoothstep(0, 0.7, d).oneMinus(), 1.6));
  return vec4(u.color.mul(u.strength).mul(u.glow).mul(falloff), shadow).mul(u.shown);
}
