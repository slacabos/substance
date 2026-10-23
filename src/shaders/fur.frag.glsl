uniform float uDensity;
uniform vec3 uRoot;
uniform vec3 uTip;
uniform vec3 uL1;
uniform mat3 uLattice;

varying vec3 vQ;
varying float vH;
varying vec3 vN;
varying vec3 vT;
varying vec3 vView;

vec3 hash33(vec3 p3) {
  p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}

void main() {
  vec3 N = normalize(vN), T = normalize(vT), V = normalize(vView), L = normalize(uL1);
  if (vH < 1e-4) {
    // skin: dark and matte, it only peeks through between strands
    vec3 skin = uRoot * 0.55 * (clamp(dot(N, L) * 0.5 + 0.5, 0.0, 1.0) + 0.15);
    gl_FragColor = vec4(skin, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
    return;
  }

  // One strand per lattice cell. The lattice is rotated so strands don't line up with cube faces.
  vec3 q = uLattice * vQ * uDensity;
  vec3 cell = floor(q);
  vec3 rnd = hash33(cell);
  vec3 ctr = cell + 0.15 + 0.7 * rnd;
  float len = 0.55 + 0.45 * fract(rnd.x * 7.13 + rnd.z * 3.1);
  float t = vH / len;
  if (t > 1.0) discard;
  float dist = length(q - ctr);
  float r = 0.46 * pow(1.0 - t, 0.6) + 0.05;
  float w = fwidth(dist) * 0.75;
  float a = 1.0 - smoothstep(r - w, r + w, dist);
  if (a < 0.02) discard;

  // Kajiya-Kay style strand shading with a soft wrap diffuse and a rim for fuzz.
  vec3 base = mix(uRoot, uTip, smoothstep(0.0, 0.95, vH)) * (0.7 + 0.55 * rnd.y);
  float ao = mix(0.15, 1.0, pow(vH, 0.85));
  float wrap = clamp(dot(N, L) * 0.6 + 0.4, 0.0, 1.0);
  float tl = dot(T, L);
  float kk = sqrt(max(0.0, 1.0 - tl * tl));
  vec3 H = normalize(L + V);
  float th = dot(T, H);
  float spec = pow(sqrt(max(0.0, 1.0 - th * th)), 80.0);
  vec3 col = base * wrap * mix(1.0, kk, 0.5) * 1.7;
  col += vec3(1.0, 0.93, 0.82) * spec * 0.3 * vH * wrap;
  float rim = pow(1.0 - max(dot(N, V), 0.0), 2.5);
  col += uTip * rim * 0.8 * vH;
  col += base * 0.22 * (0.55 + 0.45 * N.y);
  col *= ao;
  gl_FragColor = vec4(col, a);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
