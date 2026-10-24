import { Color, MathUtils, Vector2, Vector3, type PerspectiveCamera } from 'three';
import type { Node, StorageBufferNode } from 'three/webgpu';
import {
  abs,
  cameraPosition,
  clamp,
  cos,
  cross,
  dot,
  floor,
  Fn,
  instanceIndex,
  instancedArray,
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
  sin,
  smoothstep,
  sqrt,
  storage,
  transformNormalToView,
  uint,
  uniform,
  varyingProperty,
  vec3,
  vec4,
} from 'three/tsl';
import { FUR, type FurCoat } from '../core/fur';
import type { FurSim } from '../materials/furSim';
import { furGroom, furRestOffset } from './furGroom';
import { snoise } from './noise';
import { baseN, baseR } from './shape';

export function createFurUniforms(density: number = FUR.density) {
  return {
    density: uniform(density),
    root: uniform(new Color(FUR.colors.root)),
    tip: uniform(new Color(FUR.colors.tip)),
    copper: uniform(new Color(FUR.colors.copper)),
    rust: uniform(new Color(FUR.colors.rust)),
    light: uniform(new Vector3(...FUR.light).normalize()),
  };
}

export type FurUniforms = ReturnType<typeof createFurUniforms>;
type Vec3 = Node<'vec3'>;
type Float = Node<'float'>;
const bufferSize = new Vector2();

const pixelPerDistance = uniform(0).onRenderUpdate(({ camera, renderer }) => {
  const fov = MathUtils.degToRad((camera as PerspectiveCamera).fov);
  return (2 * Math.tan(fov / 2)) / renderer!.getDrawingBufferSize(bufferSize).y;
});

/** Object-space pigment variation: warm tan with copper and russet fibres, rather than separate markings. */
function coatColor(p: Vec3, rnd: Float, u: FurUniforms): Vec3 {
  const warm = sin(dot(p, vec3(4.8, 3.1, 2.4)))
    .mul(sin(dot(p, vec3(2.1, -3.6, 5.2))))
    .mul(0.5)
    .add(0.5);
  const red = sin(dot(p, vec3(-3.2, 6.1, 4.3)))
    .mul(0.5)
    .add(0.5);
  const pigment = clamp(rnd.mul(0.78).add(warm.mul(0.22)), 0, 1);
  const tan = mix(u.rust, u.tip, smoothstep(0.06, 0.62, pigment));
  return mix(mix(tan, u.root, red.mul(0.15)), u.copper, smoothstep(0.58, 1, pigment).mul(0.88)).mul(
    red.mul(0.12).add(0.94),
  );
}

/** Catmull-Rom point and derivative, with respect to the current guide segment. */
function catmullRom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, f: Float) {
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

export function createStrandNodes(u: FurUniforms, sim: FurSim, coat: FurCoat) {
  const guides = sim.renderBuffers();
  const read = (data: Float32Array) => {
    const buffer = instancedArray(data, 'vec4');
    return storage(buffer.value, 'vec4', coat.count).toReadOnly() as StorageBufferNode<'vec4'>;
  };
  const roots = read(coat.roots);
  const traits = read(coat.traits);
  const indexBuffer = instancedArray(coat.guides, 'uvec4');
  const indices = storage(indexBuffer.value, 'uvec4', coat.count).toReadOnly() as StorageBufferNode<'uvec4'>;
  const weights = read(coat.weights);
  const K = sim.points;
  const vH = varyingProperty('float', 'vFurHeight');
  const vT = varyingProperty('float', 'vFurT');
  const vThin = varyingProperty('float', 'vFurThin');
  const vPigment = varyingProperty('vec3', 'vFurPigment');
  const vN = varyingProperty('vec3', 'vFurN');
  const vTan = varyingProperty('vec3', 'vFurTangent');

  const position = Fn(() => {
    const root = roots.element(instanceIndex);
    const rnd = traits.element(instanceIndex);
    const ids = indices.element(instanceIndex);
    const blend = weights.element(instanceIndex);
    const p = root.xyz;
    const r = baseR(p);
    const q = p.mul(r);
    const localN = baseN(p, q);
    const naturalLay = furGroom(p, localN);
    const acrossLay = cross(localN, naturalLay);
    const lay = naturalLay.mul(rnd.x.mul(0.3).add(0.85)).add(acrossLay.mul(rnd.y.sub(0.5).mul(0.35)));
    const world = (point: Vec3, w: number) => sim.uniforms.anchor.mul(vec4(point, w)).xyz;
    const n = normalize(world(localN, 0));
    const t = positionGeometry.y;
    const distance = t.mul(root.w);
    // Tiny irregular bends break up straight, needle-like highlights, particularly in the woolly undercoat.
    const phase = t.mul(Math.PI * 3).add(rnd.z.mul(Math.PI * 2));
    const ripple = sin(t.mul(Math.PI)).mul(sin(phase));
    const rippleSlope = cos(t.mul(Math.PI))
      .mul(sin(phase))
      .mul(Math.PI)
      .add(
        sin(t.mul(Math.PI))
          .mul(cos(phase))
          .mul(Math.PI * 3),
      );
    const crimpAmount = rnd.w.oneMinus().mul(0.025).add(0.015);
    const crimp = acrossLay.mul(ripple).mul(root.w).mul(crimpAmount);
    const own = q.add(furRestOffset(localN, lay, t, distance)).add(crimp);

    // A small convergence towards the nearest tuft, not a shared root or a pointy lock. It is zero at roots.
    const tuftP = guides.roots.element(ids.x).xyz;
    const tuftQ = tuftP.mul(baseR(tuftP));
    const tuftN = baseN(tuftP, tuftQ);
    const tuft = tuftQ.add(furRestOffset(tuftN, furGroom(tuftP, tuftN), t, distance));
    const across = tuft.sub(own);
    const tangentOffset = across.sub(localN.mul(dot(across, localN)));
    const clumping = rnd.w.mul(0.07).add(0.06);
    const point = world(own.add(tangentOffset.mul(t.mul(t)).mul(clumping)), 1).toVar();
    const tangent = world(
      normalize(localN.add(lay.mul(t.mul(FUR.groom.bend).add(FUR.groom.rootLean)))).mul(root.w),
      0,
    ).toVar();
    tangent.addAssign(world(acrossLay.mul(rippleSlope).mul(root.w).mul(crimpAmount), 0));
    tangent.addAssign(world(tangentOffset.mul(t).mul(clumping.mul(2)), 0));

    // Short undercoat samples only the stiff bottom of a guide. Blend displacements from the uncombed rest
    // curve, never absolute guide positions: each follower stays attached to its own surface point.
    const f = distance.div(FUR.length).mul(K - 1);
    const segment = min(floor(f), K - 2);
    const i = uint(segment);
    const sample = (buffer: StorageBufferNode<'vec3'>, id: Node<'uint'>) => {
      const base = id.mul(K);
      const p1 = buffer.element(base.add(i));
      const p2 = buffer.element(base.add(i).add(1));
      const before = buffer.element(base.add(uint(max(segment, 1).sub(1))));
      const after = buffer.element(base.add(uint(min(segment.add(2), K - 1))));
      const p0 = select(i.equal(0), p1.mul(2).sub(p2), before);
      const p3 = select(i.equal(K - 2), p2.mul(2).sub(p1), after);
      return catmullRom(p0, p1, p2, p3, f.sub(segment));
    };
    for (const component of ['x', 'y', 'z'] as const) {
      const moving = sample(guides.displacement, ids[component]);
      point.addAssign(moving.point.mul(blend[component]));
      tangent.addAssign(moving.tangent.mul(blend[component]).mul(root.w.mul((K - 1) / FUR.length)));
    }

    // A blended displacement may cross a sharp cube transition; keep the visible fibre outside too.
    const local = sim.uniforms.anchorInv.mul(vec4(point, 1)).xyz;
    const radial = max(length(local), 1e-6);
    const outward = local.div(radial);
    const floorR = baseR(outward).add(t.mul(0.001));
    const safe = world(outward.mul(max(radial, floorR)), 1);
    const dir = normalize(tangent);
    const stretch = sqrt(r.mul(r).div(max(dot(p, localN), 0.3)));
    const width = pow(t.oneMinus(), 0.65)
      .mul(0.96)
      .add(0.025)
      .mul(rnd.x.mul(0.35).add(0.8))
      .mul(FUR.widthScale)
      .mul(stretch)
      .mul(sim.uniforms.scale)
      .div(u.density);
    const toCamera = cameraPosition.sub(safe);
    const drawn = max(width, pixelPerDistance.mul(length(toCamera)));
    const side0 = cross(dir, toCamera);
    const fallback = cross(n, select(abs(n.y).lessThan(0.9), vec3(0, 1, 0), vec3(1, 0, 0)));
    const side = normalize(select(length(side0).greaterThan(1e-6), side0, fallback));
    vH.assign(distance.div(FUR.length));
    vT.assign(t);
    vThin.assign(width.div(drawn));
    vPigment.assign(coatColor(p, rnd.y, u).mul(rnd.z.mul(0.22).add(0.9)));
    vN.assign(transformNormalToView(n));
    vTan.assign(normalize(modelViewMatrix.mul(vec4(dir, 0)).xyz));
    return safe.add(side.mul(positionGeometry.x).mul(drawn).mul(0.5));
  })();

  const N = normalize(vN);
  const T = normalize(vTan);
  const V = normalize(positionView.negate());
  const L = normalize(u.light);
  const base = mix(vPigment, u.tip, smoothstep(0.25, 1, vT).mul(0.1));
  const ao = mix(0.48, 1, pow(vH, 0.55));
  const wrap = clamp(dot(N, L).mul(0.55).add(0.45), 0, 1);
  const tl = dot(T, L);
  const kk = sqrt(max(tl.mul(tl).oneMinus(), 0));
  const th = dot(T, normalize(L.add(V)));
  const spec = pow(sqrt(max(th.mul(th).oneMinus(), 0)), 28);
  const rim = pow(max(dot(N, V), 0).oneMinus(), 3);
  const color = base
    .mul(wrap.mul(1.25).add(0.3))
    .mul(mix(1, kk, 0.25))
    .add(vec3(1, 0.87, 0.68).mul(spec).mul(0.07).mul(wrap).mul(vH))
    .add(u.tip.mul(rim).mul(0.24).mul(vH))
    .mul(ao);
  const opacity = vThin.mul(smoothstep(0.88, 1, vT).oneMinus());
  return { position, color, opacity };
}

/** A softly shaded, fine-grained coat base fills the gaps between the short fibres instead of black skin. */
export function createSkinNodes(u: FurUniforms) {
  const vN = varyingProperty('vec3', 'vSkinN');
  const vP = varyingProperty('vec3', 'vSkinP');
  const position = Fn(() => {
    const p = normalize(positionGeometry);
    const q = p.mul(baseR(p));
    vP.assign(p);
    vN.assign(transformNormalToView(baseN(p, q)));
    return q;
  })();
  const lit = clamp(dot(normalize(vN), normalize(u.light)).mul(0.5).add(0.5), 0, 1)
    .mul(0.65)
    .add(0.3);
  const grain = snoise(vP.mul(520)).mul(0.06).add(0.94);
  return { position, color: coatColor(vP, uniform(0.5), u).mul(lit).mul(grain).mul(0.75) };
}
