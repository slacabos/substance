// Soft pool under the object: an additive light glow on dark sets, a contact shadow on bright ones.
// Blended as premultiplied alpha, so rgb adds light and alpha takes it away.
uniform vec3 uColor;
uniform float uA;
uniform float uGlow;
uniform float uShadow;
varying vec2 vUv;

void main() {
  float d = length(vUv - 0.5) * 2.0;
  float falloff = pow(1.0 - smoothstep(0.0, 1.0, d), 3.0);
  float shadow = uShadow * pow(1.0 - smoothstep(0.0, 0.7, d), 1.6);
  gl_FragColor = vec4(uColor * uA * uGlow * falloff, shadow);
  #include <colorspace_fragment>
}
