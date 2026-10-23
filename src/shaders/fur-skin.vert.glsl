// The skin under the hair; shape.glsl is prepended.
varying vec3 vN;

void main() {
  vec3 p = normalize(position);
  vec3 q = p * baseR(p);
  vN = normalize(normalMatrix * baseN(p, q));
  gl_Position = projectionMatrix * modelViewMatrix * vec4(q, 1.0);
}
