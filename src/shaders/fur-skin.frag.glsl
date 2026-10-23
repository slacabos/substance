// Skin: dark and matte, it only peeks through between strands.
uniform vec3 uRoot;
uniform vec3 uL1;
varying vec3 vN;

void main() {
  vec3 skin = uRoot * 0.55 * (clamp(dot(normalize(vN), normalize(uL1)) * 0.5 + 0.5, 0.0, 1.0) + 0.15);
  gl_FragColor = vec4(skin, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
