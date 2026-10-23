// Injected into MeshPhysicalMaterial's vertex shader after <common>.
// MODE 0 = water, 1 = slime, 2 = ocean. MAX_IMPULSES is set through material defines.
uniform float uTime;
uniform vec4 uImp[MAX_IMPULSES];   // water/ocean: xyz direction, w start time | slime: xyz direction, w depth
uniform vec4 uImpB[MAX_IMPULSES];  // water/ocean: x strength | slime: x radius
uniform vec4 uPress;               // xyz direction, w how hard the finger is pressing
uniform vec4 uWob;                 // xyz axis, w amount of the squash-and-stretch mode

#if MODE == 2
uniform float uStorm;
varying float vWave;
varying vec3 vObj;

// sharp-crested swell: exp(sin) narrows the peaks and flattens the troughs, like wind waves
float crest(float x) { return exp(sin(x) - 1.0) - 0.466; }

float oceanAt(vec3 p) {
  float t = uTime;
  float warp = 0.5 * sin(dot(p, vec3(0.3, 0.9, 0.2)) * 3.0 + t * 0.25);
  float d = 0.0;
  d += 0.020 * crest(dot(p, normalize(vec3(1.0, 0.25, 0.35))) * 6.0 + warp - t * 1.05);
  d += 0.015 * crest(dot(p, normalize(vec3(0.6, -0.3, 1.0))) * 8.5 - warp - t * 1.3);
  d += 0.011 * crest(dot(p, normalize(vec3(-0.4, 0.8, 0.5))) * 12.0 + warp * 0.5 - t * 1.6);
  d += 0.010 * crest(dot(p, normalize(vec3(0.9, 0.5, -0.6))) * 18.0 - t * 2.0);
  d += 0.006 * crest(dot(p, normalize(vec3(-0.7, -0.2, 0.8))) * 27.0 - t * 2.5);
  d += 0.0035 * crest(dot(p, normalize(vec3(0.2, -0.9, -0.4))) * 39.0 - t * 3.1);
  d += 0.0025 * crest(dot(p, normalize(vec3(-0.5, 0.4, -0.8))) * 47.0 + warp - t * 3.4);
  return d * uStorm;
}
#endif

float arcDist(vec3 a, vec3 b) { return acos(clamp(dot(a, b), -1.0, 1.0)); }

float dispAt(vec3 p) {
  float d = 0.0;
#if MODE == 0 || MODE == 2
#if MODE == 0
  // gentle swell so the water never looks frozen
  d += 0.006 * sin(p.x * 5.0 + uTime * 1.3) * sin(p.y * 4.0 - uTime * 1.1);
  d += 0.004 * sin(p.z * 7.0 + p.x * 3.0 + uTime * 1.7);
#else
  d += oceanAt(p);
#endif
  for (int i = 0; i < MAX_IMPULSES; i++) {
    vec4 a = uImp[i];
    float str = uImpB[i].x;
    float age = uTime - a.w;
    if (str <= 0.0 || age < 0.0 || age > 5.0) continue;
    float s = arcDist(p, a.xyz);
    float behind = age * 0.85 - s;  // distance behind the wavefront
    float env = smoothstep(-0.06, 0.04, behind) * exp(-max(behind, 0.0) * 3.2) * exp(-age * 0.75) / (1.0 + 1.5 * s);
    d += str * 0.045 * env * cos(17.0 * behind);
    d -= str * 0.05 * exp(-s * s * 90.0) * exp(-age * 5.0);  // the initial splash dip
  }
  float ps = arcDist(p, uPress.xyz);
  d -= uPress.w * 0.035 * exp(-ps * ps * 70.0);
#else
  d += 0.010 * snoise(p * 2.1 + vec3(0.0, uTime * 0.18, uTime * 0.07));
  d += 0.003 * snoise(p * 5.3 - vec3(uTime * 0.12));
  for (int i = 0; i < MAX_IMPULSES; i++) {
    vec4 a = uImp[i];
    if (abs(a.w) < 1e-4) continue;
    float r = uImpB[i].x;
    float s = arcDist(p, a.xyz);
    float rim = (s - 1.7 * r) / (0.75 * r);
    d += a.w * (exp(-s * s / (r * r)) - 0.32 * exp(-rim * rim));  // dent plus the goo it shoves aside
  }
  float c = dot(p, uWob.xyz);
  d += uWob.w * (1.5 * c * c - 0.5);
#endif
  return d;
}

float orbHeight;

// Displaced position plus a finite-difference normal, so any displacement shades correctly.
void orbDeform(vec3 p, out vec3 pos, out vec3 nrm) {
  orbHeight = dispAt(p);
  pos = p * (baseR(p) + orbHeight);
  vec3 t1 = normalize(cross(p, abs(p.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  vec3 t2 = cross(p, t1);
  const float e = 0.004;
  vec3 pa = normalize(p + t1 * e);
  vec3 pb = normalize(p + t2 * e);
  vec3 A = pa * (baseR(pa) + dispAt(pa));
  vec3 B = pb * (baseR(pb) + dispAt(pb));
  nrm = normalize(cross(A - pos, B - pos));
  if (dot(nrm, p) < 0.0) nrm = -nrm;
}
