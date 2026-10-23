uniform vec3 uRoot;
uniform vec3 uTip;
uniform vec3 uL1;

varying float vH;
varying float vT;
varying float vThin;
varying float vRand;
varying vec3 vN;
varying vec3 vTan;
varying vec3 vView;

void main() {
  vec3 N = normalize(vN), T = normalize(vTan), V = normalize(vView), L = normalize(uL1);

  // Kajiya-Kay style strand shading with a soft wrap diffuse and a rim for fuzz.
  vec3 base = mix(uRoot, uTip, smoothstep(0.0, 0.95, vH)) * (0.7 + 0.55 * vRand);
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

  // Coverage: sub-pixel strands are drawn a pixel wide at matching opacity, and tips fade out.
  float a = vThin * (1.0 - smoothstep(0.8, 1.0, vT));
  gl_FragColor = vec4(col, a);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
