// Sphere-to-rounded-cube morph. Every surface point is a unit direction p scaled by baseR(p);
// at uMorph = 1 that traces the superellipsoid |x|^10 + |y|^10 + |z|^10 = 0.8^10.
uniform float uMorph;

float baseR(vec3 p) {
  vec3 a = abs(p);
  float s = pow(pow(a.x, 10.0) + pow(a.y, 10.0) + pow(a.z, 10.0), 0.1);
  return mix(1.0, 0.8 / s, uMorph);
}

// Surface normal at q = p * baseR(p), blended so it stays smooth mid-morph.
vec3 baseN(vec3 p, vec3 q) {
  vec3 g = sign(q) * pow(abs(q), vec3(9.0));
  return normalize(mix(p, normalize(g), clamp(uMorph, 0.0, 1.0)));
}
