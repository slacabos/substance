// Hair strands: one instance per strand, drawn as a camera-facing ribbon from root to tip.
// The template's position.x is the side (-1 or 1) and position.y runs 0 at the root to 1 at the tip.
// MAX_BRUSHES is set through material defines; shape.glsl is prepended.
uniform float uLen;
uniform float uDensity;
uniform float uTime;
uniform vec3 uGrav;     // gravity in object space
uniform vec3 uOmega;    // lagged spin in object space
uniform vec4 uBrush[MAX_BRUSHES];   // xyz contact direction, w strength
uniform vec4 uBrushT[MAX_BRUSHES];  // xyz stroke direction
uniform vec4 uPress;    // xyz direction, w press amount
uniform vec3 uCamera;   // camera position in object space
uniform float uPixel;   // size of one pixel per unit of distance from the camera

attribute vec3 aRoot;   // unit direction of the root
attribute vec3 aRand;   // per-strand random numbers in [0, 1)

varying float vH;       // height along the full fur length, 0 at the skin
varying float vT;       // 0 at the root, 1 at the tip of this strand
varying float vThin;    // < 1 when the strand is thinner than a pixel and is drawn wider but fainter
varying float vRand;
varying vec3 vN;
varying vec3 vTan;
varying vec3 vView;

void main() {
  vec3 p = aRoot;
  vec3 q = p * baseR(p);
  vec3 n = baseN(p, q);

  vec3 bend = uGrav * 0.32;
  bend += cross(p, uOmega) * 0.2;  // hair trails behind when the object spins
  bend += 0.045 * vec3(sin(uTime * 1.3 + p.y * 5.0), sin(uTime * 1.1 + p.z * 4.0), sin(uTime * 1.7 + p.x * 4.5));
  for (int i = 0; i < MAX_BRUSHES; i++) {
    vec4 b = uBrush[i];
    if (abs(b.w) < 1e-3) continue;
    float s = acos(clamp(dot(p, b.xyz), -1.0, 1.0));
    bend += uBrushT[i].xyz * b.w * exp(-s * s / 0.05);
  }
  vec3 away = p - uPress.xyz * dot(p, uPress.xyz);
  float al = length(away);
  float ps = acos(clamp(dot(p, uPress.xyz), -1.0, 1.0));
  if (al > 1e-4) bend += (away / al) * uPress.w * 1.3 * exp(-ps * ps / 0.035) * smoothstep(0.0, 0.08, ps);
  bend -= n * dot(bend, n);
  float bl = length(bend);
  if (bl > 1.2) {
    bend *= 1.2 / bl;
    bl = 1.2;
  }

  float t = position.y;
  float h = t * (0.55 + 0.45 * aRand.x);
  float up = uLen * (1.0 - 0.35 * bl);
  vec3 pos = q + n * h * up + bend * uLen * h * h;
  vec3 dir = normalize(n * up + 2.0 * h * bend * uLen);

  // Width tapers to a fine tip. Directions spread out where the cube's surface stretches away from the
  // sphere, so strands there get wider to keep the coat looking as dense.
  float stretch = sqrt(baseR(p) * baseR(p) / max(dot(p, n), 0.3));
  float width = (0.92 * pow(1.0 - t, 0.6) + 0.1) * (0.8 + 0.4 * aRand.y) * stretch / uDensity;
  vec3 toCamera = uCamera - pos;
  float pixel = uPixel * length(toCamera);
  float drawn = max(width, pixel);
  vThin = width / drawn;

  // Face the camera; a strand pointing straight at it just needs some sideways direction.
  vec3 side = cross(dir, toCamera);
  side = length(side) > 1e-6 ? normalize(side) : normalize(cross(n, vec3(0.31, 0.83, 0.46)));
  pos += side * position.x * drawn * 0.5;

  vec4 mv = modelViewMatrix * vec4(pos, 1.0);
  vH = h;
  vT = t;
  vRand = aRand.z;
  vN = normalize(normalMatrix * n);
  vTan = normalize(normalMatrix * dir);
  vView = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}
