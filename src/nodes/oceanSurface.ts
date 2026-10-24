import type { Node, StorageBufferNode } from 'three/webgpu';
import {
  abs,
  clamp,
  cos,
  cross,
  dot,
  float,
  floor,
  If,
  Loop,
  max,
  normalize,
  select,
  sin,
  uint,
  vec3,
} from 'three/tsl';
import { LOOKUP_SIZE } from '../core/oceanLookup';

// Shared GPU helpers for reading the interactive ocean surface (materials/oceanSim.ts) at a direction.

type Vec3 = Node<'vec3'>;
type UInt = Node<'uint'>;

/**
 * The wind's direction on the orb at unit direction p, as windAt in core/oceanWind.ts: `angle` from east toward
 * north, as long as the cosine of the latitude.
 */
export function windAt(p: Vec3, angle: Node<'float'>): Vec3 {
  const c = cos(angle);
  const s = sin(angle);
  return vec3(
    c.mul(p.z).sub(s.mul(p.x).mul(p.y)),
    s.mul(float(1).sub(p.y.mul(p.y))),
    c.mul(p.x).negate().sub(s.mul(p.z).mul(p.y)),
  );
}

/**
 * A fixed tangent basis at unit direction p, for storing a vertex's foam drift in two numbers. It jumps near the
 * y poles, which doesn't matter: each vertex only ever reads its own drift back in its own basis.
 */
export function driftBasis(p: Vec3): { e1: Vec3; e2: Vec3 } {
  const axis = select(abs(p.y).lessThan(0.9), vec3(0, 1, 0), vec3(1, 0, 0));
  const e1 = normalize(cross(axis, p));
  return { e1, e2: cross(p, e1) };
}

/**
 * The cube-map cell of a direction, matching core/oceanLookup.ts: faces +x, -x, +y, -y, +z, -z, each split
 * into size x size cells along its two other axes.
 */
export function cubeCell(d: Vec3, size: number): UInt {
  const a = abs(d);
  const onX = a.x.greaterThanEqual(a.y).and(a.x.greaterThanEqual(a.z));
  const onY = a.y.greaterThanEqual(a.z);
  const axis = select(onX, uint(0), select(onY, uint(1), uint(2)));
  const major = select(onX, d.x, select(onY, d.y, d.z));
  const face = axis.mul(2).add(select(major.greaterThanEqual(0), uint(0), uint(1)));
  // the face's own axes: x faces run along (y, z), y faces along (x, z), z faces along (x, y)
  const uv = select(onX, vec3(d.y, d.z, 0), select(onY, vec3(d.x, d.z, 0), vec3(d.x, d.y, 0))).div(
    max(abs(major), 1e-6),
  );
  const cell = (v: Node<'float'>) => uint(clamp(floor(v.add(1).mul(0.5 * size)), float(0), float(size - 1)));
  return face
    .mul(size * size)
    .add(cell(uv.y).mul(size))
    .add(cell(uv.x));
}

export interface SurfaceBuffers {
  surface: StorageBufferNode<'vec4'>;
  flow: StorageBufferNode<'vec4'>;
  ground: StorageBufferNode<'vec4'>;
  vertex: StorageBufferNode<'vec4'>;
  neighbor: StorageBufferNode<'uint'>;
  lookup: StorageBufferNode<'uint'>;
}

/**
 * The sea's own height at unit direction d (the hand's waves, without the scripted surf: OceanSim's slope.w),
 * blended over the nearest vertex and its neighbours like sampleSurface. Four storage buffers.
 */
export function seaHeightAt(
  d: Vec3,
  buffers: Pick<SurfaceBuffers, 'lookup' | 'vertex' | 'neighbor'> & { slope: StorageBufferNode<'vec4'> },
): Node<'float'> {
  const centre = buffers.lookup.element(cubeCell(d, LOOKUP_SIZE));
  const height = float(0).toVar();
  const weight = float(0).toVar();
  const add = (i: UInt) => {
    const offset = buffers.vertex.element(i).xyz.sub(d);
    const w = float(1).div(dot(offset, offset).add(0.0002));
    height.addAssign(buffers.slope.element(i).w.mul(w));
    weight.addAssign(w);
  };
  add(centre);
  Loop({ start: 0, end: 6, type: 'uint' }, ({ i: k }) => {
    const j = buffers.neighbor.element(centre.mul(6).add(k));
    If(j.notEqual(centre), () => add(j));
  });
  return height.div(weight);
}

/**
 * The surface at unit direction d: the nearest vertex from the lookup, blended with its six neighbours by
 * inverse squared distance. Returns height (with surf), foam, the current, and distance to the shore.
 */
export function sampleSurface(
  d: Vec3,
  buffers: SurfaceBuffers,
): { height: Node<'float'>; foam: Node<'float'>; flow: Vec3; shore: Node<'float'> } {
  const centre = buffers.lookup.element(cubeCell(d, LOOKUP_SIZE));
  const height = float(0).toVar();
  const foam = float(0).toVar();
  const flow = vec3(0).toVar();
  const weight = float(0).toVar();
  const add = (i: UInt) => {
    const offset = buffers.vertex.element(i).xyz.sub(d);
    const w = float(1).div(dot(offset, offset).add(0.0002));
    const s = buffers.surface.element(i);
    height.addAssign(s.x.mul(w));
    foam.addAssign(s.y.mul(w));
    flow.addAssign(buffers.flow.element(i).xyz.mul(w));
    weight.addAssign(w);
  };
  add(centre);
  Loop({ start: 0, end: 6, type: 'uint' }, ({ i: k }) => {
    const j = buffers.neighbor.element(centre.mul(6).add(k));
    If(j.notEqual(centre), () => add(j));
  });
  return {
    height: height.div(weight),
    foam: foam.div(weight),
    flow: flow.div(weight),
    shore: buffers.ground.element(centre).y,
  };
}
