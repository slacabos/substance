// Studio sweep: a wall-to-floor gradient, a band where they meet and a soft light behind the object.
uniform vec3 uWall;
uniform vec3 uFloor;
uniform vec3 uHorizon;
uniform vec3 uHalo;
varying vec3 vDir;

void main() {
  vec3 d = normalize(vDir);
  float az = atan(d.x, -d.z);
  vec3 col = mix(uFloor, uWall, smoothstep(-0.5, 0.8, d.y));
  col += uHorizon * exp(-pow(d.y + 0.08, 2.0) * 25.0);
  col += uHalo * exp(-pow(az * 1.3, 2.0) - pow((d.y - 0.8) * 3.0, 2.0));
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
