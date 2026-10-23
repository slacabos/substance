import { Color, MathUtils, Vector2, Vector3, type PerspectiveCamera } from 'three';
import type { Node } from 'three/webgpu';
import {
  cameraPosition,
  clamp,
  cross,
  dot,
  floor,
  Fn,
  instanceIndex,
  length,
  max,
  min,
  mix,
  modelViewMatrix,
  normalize,
  positionGeometry,
  positionView,
  pow,
  select,
  smoothstep,
  sqrt,
  transformNormalToView,
  uint,
  uniform,
  varyingProperty,
  vec3,
  vec4,
} from 'three/tsl';
import { FUR } from '../config';
import type { FurSim } from '../materials/furSim';
import { baseN, baseR } from './shape';

// Fur: camera-facing strand ribbons over a dark skin. Strand template: position.x is the side (-1 or 1),
// position.y runs 0 at the root to 1 at the tip. Each strand follows the points FurSim simulates.

export function createFurUniforms() {
  return {
    density: uniform(FUR.density),
    root: uniform(new Color('#2a1107')),
    tip: uniform(new Color('#f3b97f')),
    /** Key light direction, in view space. */
    light: uniform(new Vector3(-0.55, 0.75, 0.5).normalize()),
  };
}

export type FurUniforms = ReturnType<typeof createFurUniforms>;

type Vec3 = Node<'vec3'>;

const bufferSize = new Vector2();

/** How wide one pixel is per unit of distance from the camera, refreshed every render. */
const pixelPerDistance = uniform(0).onRenderUpdate(({ camera, renderer }) => {
  const fov = MathUtils.degToRad((camera as PerspectiveCamera).fov);
  return (2 * Math.tan(fov / 2)) / renderer!.getDrawingBufferSize(bufferSize).y;
});

/** Uniform Catmull-Rom spline through p1..p2 (p0 and p3 shape it), at fraction f: position and tangent. */
function catmullRom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, f: Node<'float'>) {
  const a = p2.sub(p0);
  const b = p0.mul(2).sub(p1.mul(5)).add(p2.mul(4)).sub(p3);
  const c = p1.mul(3).sub(p0).sub(p2.mul(3)).add(p3);
  const point = p1
    .mul(2)
    .add(a.mul(f))
    .add(b.mul(f.mul(f)))
    .add(c.mul(f.mul(f).mul(f)))
    .mul(0.5);
  const tangent = a
    .add(b.mul(f.mul(2)))
    .add(c.mul(f.mul(f).mul(3)))
    .mul(0.5);
  return { point, tangent };
}

export function createStrandNodes(u: FurUniforms, sim: FurSim) {
  const { roots, rand, pos } = sim.renderBuffers();
  const K = sim.points;
  const vH = varyingProperty('float', 'vH'); // height along the full fur length, 0 at the skin
  const vT = varyingProperty('float', 'vT'); // 0 at the root, 1 at the tip of this strand
  const vThin = varyingProperty('float', 'vThin'); // < 1 when drawn wider than the strand, and fainter
  const vRand = varyingProperty('float', 'vRand');
  const vN = varyingProperty('vec3', 'vN');
  const vTan = varyingProperty('vec3', 'vTan');

  // The strands are in world space (the mesh sits at the scene root), following the simulated points.
  const position = Fn(() => {
    const p = roots.element(instanceIndex);
    const rnd = rand.element(instanceIndex);
    const n = normalize(sim.uniforms.anchor.mul(vec4(baseN(p, p.mul(baseR(p))), 0)).xyz);

    const t = positionGeometry.y;
    const f = t.mul(K - 1);
    const segment = min(floor(f), K - 2);
    const i = uint(segment);
    const base = instanceIndex.mul(K);
    const p1 = pos.element(base.add(i));
    const p2 = pos.element(base.add(i).add(1));
    // past either end, extend the first or last segment straight
    const before = pos.element(base.add(uint(max(segment, 1).sub(1))));
    const after = pos.element(base.add(uint(min(segment.add(2), K - 1))));
    const p0 = select(i.equal(0), p1.mul(2).sub(p2), before);
    const p3 = select(i.equal(K - 2), p2.mul(2).sub(p1), after);
    const { point, tangent } = catmullRom(p0, p1, p2, p3, f.sub(segment));
    const dir = normalize(tangent);

    // Width tapers to a fine tip. Directions spread out where the cube's surface stretches away from the
    // sphere, so strands there get wider to keep the coat looking as dense.
    const r = baseR(p);
    const stretch = sqrt(r.mul(r).div(max(dot(p, baseN(p, p.mul(r))), 0.3)));
    const width = pow(t.oneMinus(), 0.6)
      .mul(0.92)
      .add(0.1)
      .mul(rnd.y.mul(0.4).add(0.8))
      .mul(stretch)
      .mul(sim.uniforms.scale)
      .div(u.density);
    // Never thinner than a pixel.
    const toCamera = cameraPosition.sub(point);
    const drawn = max(width, pixelPerDistance.mul(length(toCamera)));

    // Face the camera; a strand pointing straight at it just needs some sideways direction.
    const side0 = cross(dir, toCamera);
    const side = select(
      length(side0).greaterThan(1e-6),
      normalize(side0),
      normalize(cross(n, vec3(0.31, 0.83, 0.46))),
    );

    vH.assign(t.mul(rnd.x.mul(0.45).add(0.55)));
    vT.assign(t);
    vThin.assign(width.div(drawn));
    vRand.assign(rnd.z);
    vN.assign(transformNormalToView(n));
    vTan.assign(normalize(modelViewMatrix.mul(vec4(dir, 0)).xyz));
    return point.add(side.mul(positionGeometry.x).mul(drawn).mul(0.5));
  })();

  // Kajiya-Kay style strand shading with a soft wrap diffuse and a rim for fuzz.
  const N = normalize(vN);
  const T = normalize(vTan);
  const V = normalize(positionView.negate());
  const L = normalize(u.light);
  const base = mix(u.root, u.tip, smoothstep(0, 0.95, vH)).mul(vRand.mul(0.55).add(0.7));
  const ao = mix(0.15, 1, pow(vH, 0.85));
  const wrap = clamp(dot(N, L).mul(0.6).add(0.4), 0, 1);
  const tl = dot(T, L);
  const kk = sqrt(max(tl.mul(tl).oneMinus(), 0));
  const th = dot(T, normalize(L.add(V)));
  const spec = pow(sqrt(max(th.mul(th).oneMinus(), 0)), 80);
  const rim = pow(max(dot(N, V), 0).oneMinus(), 2.5);
  const color = base
    .mul(wrap)
    .mul(mix(1, kk, 0.5))
    .mul(1.7)
    .add(vec3(1, 0.93, 0.82).mul(spec).mul(0.3).mul(vH).mul(wrap))
    .add(u.tip.mul(rim).mul(0.8).mul(vH))
    .add(base.mul(0.22).mul(N.y.mul(0.45).add(0.55)))
    .mul(ao);
  // Coverage: sub-pixel strands are drawn a pixel wide at matching opacity, and tips fade out.
  const opacity = vThin.mul(smoothstep(0.8, 1, vT).oneMinus());

  return { position, color, opacity };
}

/** The skin under the hair: dark and matte, it only peeks through between strands. */
export function createSkinNodes(u: FurUniforms) {
  const vN = varyingProperty('vec3', 'vSkinN');
  const position = Fn(() => {
    const p = normalize(positionGeometry);
    const q = p.mul(baseR(p));
    vN.assign(transformNormalToView(baseN(p, q)));
    return q;
  })();
  const lit = clamp(dot(normalize(vN), normalize(u.light)).mul(0.5).add(0.5), 0, 1).add(0.15);
  return { position, color: u.root.mul(0.55).mul(lit) };
}
