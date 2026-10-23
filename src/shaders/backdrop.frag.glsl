// Dark studio with a few light strips. Mostly seen refracted through the water.
varying vec3 vDir;

float strip(float az, float y, float c, float w, float y0, float y1) {
  float a = 1.0 - smoothstep(w * 0.5, w * 0.5 + 0.025, abs(az - c));
  float b = smoothstep(y0 - 0.06, y0 + 0.06, y) * (1.0 - smoothstep(y1 - 0.06, y1 + 0.06, y));
  return a * b;
}

void main() {
  vec3 d = normalize(vDir);
  float az = atan(d.x, -d.z);
  vec3 col = mix(vec3(0.006, 0.008, 0.016), vec3(0.040, 0.050, 0.085), smoothstep(-0.5, 0.8, d.y));
  col += vec3(0.035, 0.050, 0.090) * exp(-pow(d.y + 0.08, 2.0) * 25.0);
  col += vec3(1.5, 1.6, 1.8) * strip(az, d.y, -0.46, 0.08, -0.20, 0.60);
  col += vec3(1.8, 1.3, 0.9) * strip(az, d.y, 0.52, 0.05, -0.05, 0.48);
  col += vec3(0.7, 0.8, 1.0) * 0.7 * exp(-pow(az * 1.3, 2.0) - pow((d.y - 0.8) * 3.0, 2.0));
  col += vec3(0.3, 0.5, 0.7) * 0.25 * exp(-pow((az - 3.14159) * 1.2, 2.0) - pow(d.y * 2.5, 2.0));
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
