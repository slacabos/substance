import { Color, MathUtils, Vector2, Vector3, Vector4, type PerspectiveCamera } from 'three';
import type { Node, UniformArrayNode } from 'three/webgpu';
import {
  abs,
  acos,
  attribute,
  cameraPosition,
  clamp,
  Continue,
  cross,
  dot,
  exp,
  float,
  Fn,
  If,
  length,
  Loop,
  max,
  mix,
  modelViewMatrix,
  modelWorldMatrixInverse,
  normalize,
  positionGeometry,
  positionView,
  pow,
  select,
  sin,
  smoothstep,
  sqrt,
  transformNormalToView,
  uniform,
  uniformArray,
  varyingProperty,
  vec3,
  vec4,
} from 'three/tsl';
import { FUR, MAX_BRUSHES } from '../config';
import { baseN, baseR } from './shape';

// Fur: camera-facing strand ribbons over a dark skin. Strand template: position.x is the side (-1 or 1),
// position.y runs 0 at the root to 1 at the tip; per-strand aRoot (unit root direction) and aRand.

export function createFurUniforms() {
  const brushes = Array.from({ length: MAX_BRUSHES }, () => new Vector4(0, 1, 0, 0));
  const strokes = Array.from({ length: MAX_BRUSHES }, () => new Vector4());
  return {
    len: uniform(FUR.length),
    density: uniform(FUR.density),
    time: uniform(0),
    /** Gravity in object space. */
    grav: uniform(new Vector3(0, -1, 0)),
    /** Lagged spin in object space. */
    omega: uniform(new Vector3()),
    /** xyz contact direction, w strength. */
    brush: uniformArray(brushes, 'vec4') as UniformArrayNode<'vec4'>,
    /** xyz stroke direction. */
    brushT: uniformArray(strokes, 'vec4') as UniformArrayNode<'vec4'>,
    /** The arrays behind brush and brushT: write to these, they upload every frame. */
    brushes,
    strokes,
    /** xyz direction, w press amount. */
    press: uniform(new Vector4(0, 0, 1, 0)),
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

/** How far the hair bends: gravity, spin trailing, a gentle sway, brush strokes and the press parting. */
function hairBend(p: Vec3, n: Vec3, u: FurUniforms) {
  const t = u.time;
  const bend = u.grav.mul(0.32).toVar();
  bend.addAssign(cross(p, u.omega).mul(0.2)); // hair trails behind when the object spins
  bend.addAssign(
    vec3(
      sin(t.mul(1.3).add(p.y.mul(5))),
      sin(t.mul(1.1).add(p.z.mul(4))),
      sin(t.mul(1.7).add(p.x.mul(4.5))),
    ).mul(0.045),
  );
  Loop(MAX_BRUSHES, ({ i }) => {
    const b = u.brush.element(i);
    If(abs(b.w).lessThan(1e-3), () => {
      Continue();
    });
    const s = acos(clamp(dot(p, b.xyz), -1, 1));
    bend.addAssign(
      u.brushT
        .element(i)
        .xyz.mul(b.w)
        .mul(exp(s.mul(s).div(-0.05))),
    );
  });
  const away = p.sub(u.press.xyz.mul(dot(p, u.press.xyz)));
  const al = length(away);
  const ps = acos(clamp(dot(p, u.press.xyz), -1, 1));
  If(al.greaterThan(1e-4), () => {
    bend.addAssign(
      away
        .div(al)
        .mul(u.press.w.mul(1.3))
        .mul(exp(ps.mul(ps).div(-0.035)))
        .mul(smoothstep(0, 0.08, ps)),
    );
  });
  bend.subAssign(n.mul(dot(bend, n)));
  const bl = length(bend).toVar();
  If(bl.greaterThan(1.2), () => {
    bend.mulAssign(float(1.2).div(bl));
    bl.assign(1.2);
  });
  return { bend, bl };
}

export function createStrandNodes(u: FurUniforms) {
  const vH = varyingProperty('float', 'vH'); // height along the full fur length, 0 at the skin
  const vT = varyingProperty('float', 'vT'); // 0 at the root, 1 at the tip of this strand
  const vThin = varyingProperty('float', 'vThin'); // < 1 when drawn wider than the strand, and fainter
  const vRand = varyingProperty('float', 'vRand');
  const vN = varyingProperty('vec3', 'vN');
  const vTan = varyingProperty('vec3', 'vTan');

  const position = Fn(() => {
    const p = attribute<'vec3'>('aRoot', 'vec3');
    const rnd = attribute<'vec3'>('aRand', 'vec3');
    const q = p.mul(baseR(p));
    const n = baseN(p, q);
    const { bend, bl } = hairBend(p, n, u);

    const t = positionGeometry.y;
    const h = t.mul(rnd.x.mul(0.45).add(0.55));
    const up = u.len.mul(bl.mul(-0.35).add(1));
    const pos = q.add(n.mul(h).mul(up)).add(bend.mul(u.len).mul(h).mul(h));
    const dir = normalize(n.mul(up).add(bend.mul(h.mul(2).mul(u.len))));

    // Width tapers to a fine tip. Directions spread out where the cube's surface stretches away from the
    // sphere, so strands there get wider to keep the coat looking as dense.
    const r = baseR(p);
    const stretch = sqrt(r.mul(r).div(max(dot(p, n), 0.3)));
    const width = pow(t.oneMinus(), 0.6)
      .mul(0.92)
      .add(0.1)
      .mul(rnd.y.mul(0.4).add(0.8))
      .mul(stretch)
      .div(u.density);
    // Never thinner than a pixel.
    const toCamera = modelWorldMatrixInverse.mul(vec4(cameraPosition, 1)).xyz.sub(pos);
    const pixel = pixelPerDistance.mul(length(toCamera));
    const drawn = max(width, pixel);

    // Face the camera; a strand pointing straight at it just needs some sideways direction.
    const side0 = cross(dir, toCamera);
    const side = select(
      length(side0).greaterThan(1e-6),
      normalize(side0),
      normalize(cross(n, vec3(0.31, 0.83, 0.46))),
    );

    vH.assign(h);
    vT.assign(t);
    vThin.assign(width.div(drawn));
    vRand.assign(rnd.z);
    vN.assign(transformNormalToView(n));
    vTan.assign(normalize(modelViewMatrix.mul(vec4(dir, 0)).xyz));
    return pos.add(side.mul(positionGeometry.x).mul(drawn).mul(0.5));
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
