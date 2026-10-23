// Shell-textured fur: one instance per shell, instance 0 is the skin.
// MAX_BRUSHES is set through material defines; shape.glsl is prepended.
uniform float uShells;
uniform float uLen;
uniform float uTime;
uniform vec3 uGrav;     // gravity in object space
uniform vec3 uOmega;    // lagged spin in object space
uniform vec4 uBrush[MAX_BRUSHES];   // xyz contact direction, w strength
uniform vec4 uBrushT[MAX_BRUSHES];  // xyz stroke direction
uniform vec4 uPress;    // xyz direction, w press amount

varying vec3 vQ;
varying float vH;
varying vec3 vN;
varying vec3 vT;
varying vec3 vView;

void main() {
  float h = float(gl_InstanceID) / uShells;
  vec3 p = normalize(position);
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
  float up = uLen * (1.0 - 0.35 * bl);
  vec3 pos = q + n * h * up + bend * uLen * h * h;
  vec3 dir = n * up + 2.0 * h * bend * uLen;

  vec4 mv = modelViewMatrix * vec4(pos, 1.0);
  vQ = q;
  vH = h;
  vView = -mv.xyz;
  vN = normalize(normalMatrix * n);
  vT = normalize(normalMatrix * dir);
  gl_Position = projectionMatrix * mv;
}
