import {
  Color,
  HalfFloatType,
  LinearFilter,
  Matrix4,
  PerspectiveCamera,
  RenderTarget,
  Vector2,
  Vector3,
  type Scene,
  type Texture,
} from 'three';
import {
  MeshBasicNodeMaterial,
  QuadMesh,
  type Node,
  type TextureNode,
  type WebGPURenderer,
} from 'three/webgpu';
import {
  cross,
  dot,
  exp,
  float,
  Fn,
  getViewPosition,
  If,
  max,
  mix,
  normalize,
  pmremTexture,
  pow,
  reflect,
  screenUV,
  select,
  smoothstep,
  texture,
  uniform,
  vec2,
  vec3,
  vec4,
  viewZToPerspectiveDepth,
} from 'three/tsl';

/** Camera layer that only the fluid's depth pass draws: dense splash water as sphere impostors. */
export const FLUID_LAYER = 1;

/** The depth targets are this fraction of the canvas; smoothing hides the lower resolution. */
const SCALE = 0.5;
/**
 * Smoothing: two 2D passes of a 7 x 7 kernel, the second with its taps twice as far apart (à trous), reaching
 * about nine texels. A separable filter would be cheaper, but with depth in the weights its two directions no
 * longer multiply out, and splashes come out with straight, axis-aligned edges.
 */
const BLUR_RADIUS = 3;
const BLUR_SIGMA = 2;
/** Depth sigma in view units, about a droplet: farther neighbours are separate sheets of water. */
const DEPTH_SIGMA = 0.02;

type Vec3 = Node<'vec3'>;

/**
 * Screen-space fluid rendering (van der Laan et al. 2009) for splashes: dense spray is drawn as sphere
 * impostors into a depth target (FLUID_LAYER), the depth is smoothed with a bilateral filter that keeps
 * separate sheets apart, and a full-screen composite shades it as one water surface: normals from the smoothed
 * depth, Fresnel reflection of the ocean's sky, the scene behind refracted through it, a water tint, aeration
 * and the sun's glint. Isolated droplets and mist are drawn in the main pass instead (materials/oceanFluid.ts).
 */
export class FluidPass {
  private readonly depth: RenderTarget;
  private readonly blurA: RenderTarget;
  private readonly blurB: RenderTarget;
  private readonly camera = new PerspectiveCamera();
  private readonly blurNear: QuadMesh;
  private readonly blurFar: QuadMesh;
  private readonly size = new Vector2();
  // the scene camera, for the full-screen composite (whose own camera is the quad's)
  private readonly projectionInverse = uniform(new Matrix4());
  private readonly viewToWorld = uniform(new Matrix4());
  private readonly near = uniform(0.1);
  private readonly far = uniform(100);
  private readonly texel = uniform(new Vector2(1, 1));
  private readonly sunDirection = uniform(new Vector3(0, 1, 0));
  private readonly sunColor = uniform(new Color(0, 0, 0));

  constructor() {
    const target = () => {
      // half floats always filter (32-bit ones only with an optional feature), so the half-resolution
      // layer magnifies smoothly
      const t = new RenderTarget(1, 1, { type: HalfFloatType, depthBuffer: false });
      t.texture.minFilter = t.texture.magFilter = LinearFilter;
      return t;
    };
    this.depth = new RenderTarget(1, 1, { type: HalfFloatType });
    this.blurA = target();
    this.blurB = target();
    this.camera.layers.set(FLUID_LAYER);
    this.blurNear = new QuadMesh(this.blurMaterial(this.depth.texture, 1));
    this.blurFar = new QuadMesh(this.blurMaterial(this.blurA.texture, 2));
  }

  setSun(direction: Vector3, color: Color): void {
    this.sunDirection.value.copy(direction);
    this.sunColor.value.copy(color);
  }

  /** Draws and smooths the splash depth for this frame; call before the pipeline that composites it. */
  render(renderer: WebGPURenderer, scene: Scene, camera: PerspectiveCamera): void {
    renderer.getDrawingBufferSize(this.size);
    const width = Math.max(1, Math.round(this.size.x * SCALE));
    const height = Math.max(1, Math.round(this.size.y * SCALE));
    if (this.depth.width !== width || this.depth.height !== height) {
      for (const t of [this.depth, this.blurA, this.blurB]) t.setSize(width, height);
      this.texel.value.set(1 / width, 1 / height);
    }
    this.camera.copy(camera, false);
    this.camera.layers.set(FLUID_LAYER);
    this.projectionInverse.value.copy(camera.projectionMatrixInverse);
    this.viewToWorld.value.copy(camera.matrixWorld);
    this.near.value = camera.near;
    this.far.value = camera.far;

    const previous = renderer.getRenderTarget();
    const clear = renderer.getClearColor(new Color());
    const alpha = renderer.getClearAlpha();
    renderer.setClearColor(0x000000, 0);
    renderer.setRenderTarget(this.depth);
    renderer.clear();
    renderer.render(scene, this.camera);
    renderer.setRenderTarget(this.blurA);
    this.blurNear.render(renderer);
    renderer.setRenderTarget(this.blurB);
    this.blurFar.render(renderer);
    renderer.setRenderTarget(previous);
    renderer.setClearColor(clear, alpha);
  }

  /**
   * One pass of the smoothing, its taps `spacing` texels apart: a bilateral filter on view distance (x),
   * weighted by coverage (y), so it neither blends a near sheet into a far one nor pulls the empty background
   * in; coverage itself spreads a little, softening the edges.
   */
  private blurMaterial(source: Texture, spacing: number): MeshBasicNodeMaterial {
    const material = new MeshBasicNodeMaterial();
    material.colorNode = Fn(() => {
      const step = this.texel.mul(spacing);
      const centre = texture(source, screenUV);
      const depth = float(0).toVar();
      const cover = float(0).toVar();
      const total = float(0).toVar();
      for (let i = -BLUR_RADIUS; i <= BLUR_RADIUS; i++)
        for (let j = -BLUR_RADIUS; j <= BLUR_RADIUS; j++) {
          const s = texture(source, screenUV.add(step.mul(vec2(i, j))));
          const spatial = Math.exp(-(i * i + j * j) / (2 * BLUR_SIGMA * BLUR_SIGMA));
          // an empty centre takes any neighbour; a covered one only those at about its own distance
          const gap = s.x.sub(centre.x);
          const range = select(
            centre.y.greaterThan(0),
            exp(gap.mul(gap).mul(-0.5 / (DEPTH_SIGMA * DEPTH_SIGMA))),
            1,
          );
          const w = range.mul(spatial);
          depth.addAssign(s.x.mul(s.y).mul(w));
          cover.addAssign(s.y.mul(w));
          total.addAssign(w);
        }
      return vec4(depth.div(max(cover, 1e-6)), cover.div(total), 0, 1);
    })();
    return material;
  }

  /**
   * The composite: `scene` is the scene pass's colour and `sceneViewZ` its view-space depth; `environment` the
   * ocean's sky reflection (PMREM).
   */
  composite(scene: TextureNode, sceneViewZ: Node<'float'>, environment: Texture): Node<'vec4'> {
    return Fn(() => {
      const fluid = this.blurB.texture;
      const sample = (uv: Node<'vec2'>) => texture(fluid, uv);
      const here = sample(screenUV);
      const background = scene.sample(screenUV);
      const out = vec4(background).toVar();
      // in front of whatever the scene drew there
      const visible = smoothstep(0.25, 0.6, here.y).mul(
        select(here.x.lessThan(sceneViewZ.negate().add(0.004)), float(1), float(0)),
      );
      If(visible.greaterThan(0.001), () => {
        const at = (uv: Node<'vec2'>, distance: Node<'float'>): Vec3 =>
          getViewPosition(
            uv,
            viewZToPerspectiveDepth(distance.negate(), this.near, this.far),
            this.projectionInverse,
          );
        const p = at(screenUV, here.x);
        // each side's slope from whichever neighbour is closer in depth and covered (edges otherwise read the
        // empty background as a point at the camera); where neither is, the surface faces the camera
        const side = (step: Node<'vec2'>) => {
          const a = sample(screenUV.add(step));
          const b = sample(screenUV.sub(step));
          const da = at(screenUV.add(step), a.x).sub(p);
          const db = p.sub(at(screenUV.sub(step), b.x));
          const useA = a.y.greaterThan(0.2).and(b.y.lessThanEqual(0.2).or(da.z.abs().lessThan(db.z.abs())));
          const either = a.y.greaterThan(0.2).or(b.y.greaterThan(0.2));
          return { d: select(useA, da, db), ok: either };
        };
        const sx = side(vec2(this.texel.x, 0));
        const sy = side(vec2(0, this.texel.y));
        const raw = normalize(cross(sx.d, sy.d));
        const n = select(sx.ok.and(sy.ok), select(raw.z.lessThan(0), raw.negate(), raw), vec3(0, 0, 1));
        const v = normalize(p.negate());
        const facing = max(dot(n, v), 0);
        const fresnel = pow(float(1).sub(facing), 5).mul(0.98).add(0.02);
        const reflected = (this.viewToWorld.mul(vec4(reflect(v.negate(), n), 0)) as Node<'vec4'>).xyz;
        const sky = pmremTexture(environment, reflected, float(0.05));
        // the scene behind, bent through the water; thin edges (low coverage) let more of it through, thick
        // clumps are tinted and aerated (splash water is full of air)
        const thickness = smoothstep(0.3, 1, here.y);
        const behind = scene.sample(screenUV.add(n.xy.mul(thickness.mul(0.02)))).rgb;
        const tinted = behind.mul(mix(vec3(0.95, 0.98, 1), vec3(0.62, 0.82, 0.86), thickness));
        const water = mix(tinted, vec3(0.8, 0.88, 0.92), thickness.mul(0.35));
        const sun = normalize(
          (this.viewToWorld.transpose().mul(vec4(this.sunDirection, 0)) as Node<'vec4'>).xyz,
        );
        const glint = pow(max(dot(n, normalize(sun.add(v))), 0), 300).mul(3);
        const shaded = mix(water, sky.rgb, fresnel).add(this.sunColor.mul(glint));
        out.assign(vec4(mix(background.rgb, shaded, visible), background.a));
      });
      return out;
    })();
  }
}
