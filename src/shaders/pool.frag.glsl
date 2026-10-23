// Soft light pool under the object, drawn additively.
uniform vec3 uColor;
uniform float uA;
varying vec2 vUv;

void main() {
  float d = length(vUv - 0.5) * 2.0;
  float a = uA * pow(1.0 - smoothstep(0.0, 1.0, d), 3.0);
  gl_FragColor = vec4(uColor * a, 1.0);
  #include <colorspace_fragment>
}
