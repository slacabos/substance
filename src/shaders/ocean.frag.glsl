// Injected into the ocean's fragment shader after <common>: water colour by wave height, foam on crests.
uniform float uTime;
uniform vec3 uDeep;
uniform vec3 uShallow;
uniform vec3 uScatter;
varying float vWave;
varying vec3 vObj;

float h31(vec3 p) {
  p = fract(p * 0.3183099 + 0.1);
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float vnoise(vec3 x) {
  vec3 i = floor(x), f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(h31(i), h31(i + vec3(1, 0, 0)), f.x), mix(h31(i + vec3(0, 1, 0)), h31(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(h31(i + vec3(0, 0, 1)), h31(i + vec3(1, 0, 1)), f.x), mix(h31(i + vec3(0, 1, 1)), h31(i + vec3(1, 1, 1)), f.x), f.y),
             f.z);
}

float oceanFoam() {
  vec3 q = vObj * 38.0 + vec3(0.0, uTime * 0.35, 0.0);
  float n = vnoise(q) * 0.6 + vnoise(q * 2.7) * 0.4;
  float lace = smoothstep(0.42, 0.62, n);
  return clamp(smoothstep(0.02, 0.045, vWave + 0.018 * (n - 0.5)) * mix(0.55, 1.0, lace), 0.0, 1.0);
}

vec3 oceanColor(float foam) {
  vec3 water = mix(uDeep, uShallow, smoothstep(-0.03, 0.045, vWave));
  return mix(water, vec3(0.92, 0.96, 0.98), foam);
}
