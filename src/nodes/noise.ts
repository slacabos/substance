import type { Node } from 'three/webgpu';
import { abs, dot, float, floor, Fn, max, min, step, vec2, vec3, vec4 } from 'three/tsl';

// 3D simplex noise (Ashima Arts / Stefan Gustavson, MIT), ported line for line from GLSL. Built as one WGSL
// function that every use calls, rather than inlined at each: inlined, each call's thirty-odd locals were copied
// into the shader again, and the coast's water, which calls it a dozen times, went over the 8 KB of local
// variables Safari's WebGPU allows a shader.

const mod289v3 = (x: Node<'vec3'>) => x.sub(floor(x.mul(1 / 289)).mul(289));
const mod289v4 = (x: Node<'vec4'>) => x.sub(floor(x.mul(1 / 289)).mul(289));
const permute = (x: Node<'vec4'>) => mod289v4(x.mul(34).add(1).mul(x));
const taylorInvSqrt = (r: Node<'vec4'>) => float(1.79284291400159).sub(r.mul(0.85373472095314));

const simplex = Fn(([v]: [Node<'vec3'>]) => {
  const C = vec2(1 / 6, 1 / 3);
  const D = vec4(0, 0.5, 1, 2);
  const i0 = floor(v.add(dot(v, C.yyy)));
  const x0 = v.sub(i0).add(dot(i0, C.xxx));
  const g = step(x0.yzx, x0.xyz);
  const l = float(1).sub(g);
  const i1 = min(g.xyz, l.zxy);
  const i2 = max(g.xyz, l.zxy);
  const x1 = x0.sub(i1).add(C.xxx);
  const x2 = x0.sub(i2).add(C.yyy);
  const x3 = x0.sub(D.yyy);
  const i = mod289v3(i0);
  const p = permute(
    permute(
      permute(i.z.add(vec4(0, i1.z, i2.z, 1)))
        .add(i.y)
        .add(vec4(0, i1.y, i2.y, 1)),
    )
      .add(i.x)
      .add(vec4(0, i1.x, i2.x, 1)),
  );
  const ns = float(0.142857142857).mul(D.wyz).sub(D.xzx);
  const j = p.sub(floor(p.mul(ns.z).mul(ns.z)).mul(49));
  const x_ = floor(j.mul(ns.z));
  const y_ = floor(j.sub(x_.mul(7)));
  const x = x_.mul(ns.x).add(ns.yyyy);
  const y = y_.mul(ns.x).add(ns.yyyy);
  const h = float(1).sub(abs(x)).sub(abs(y));
  const b0 = vec4(x.xy, y.xy);
  const b1 = vec4(x.zw, y.zw);
  const s0 = floor(b0).mul(2).add(1);
  const s1 = floor(b1).mul(2).add(1);
  const sh = step(h, vec4(0)).negate();
  const a0 = b0.xzyw.add(s0.xzyw.mul(sh.xxyy));
  const a1 = b1.xzyw.add(s1.xzyw.mul(sh.zzww));
  const norm = taylorInvSqrt(
    vec4(
      dot(vec3(a0.xy, h.x), vec3(a0.xy, h.x)),
      dot(vec3(a0.zw, h.y), vec3(a0.zw, h.y)),
      dot(vec3(a1.xy, h.z), vec3(a1.xy, h.z)),
      dot(vec3(a1.zw, h.w), vec3(a1.zw, h.w)),
    ),
  );
  const p0 = vec3(a0.xy, h.x).mul(norm.x);
  const p1 = vec3(a0.zw, h.y).mul(norm.y);
  const p2 = vec3(a1.xy, h.z).mul(norm.z);
  const p3 = vec3(a1.zw, h.w).mul(norm.w);
  const m0 = max(float(0.6).sub(vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3))), 0);
  const m = m0.mul(m0);
  return dot(m.mul(m), vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3))).mul(42);
}).setLayout({ name: 'snoise', type: 'float', inputs: [{ name: 'v', type: 'vec3' }] });

export function snoise(v: Node<'vec3'>): Node<'float'> {
  return simplex(v) as Node<'float'>;
}
