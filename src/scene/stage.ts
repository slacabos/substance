import { ACESFilmicToneMapping, Group, MathUtils, PerspectiveCamera, Scene } from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { PMREMGenerator, WebGPURenderer } from 'three/webgpu';
import { FIT_RADIUS, SUPERSAMPLE_PIXELS } from '../config';
import { Backdrop } from './backdrop';
import type { SettingId } from './settings';

type GpuNavigator = Navigator & { gpu?: { requestAdapter(): Promise<unknown> } };

/** Renderer, camera, lighting environment and the spinning anchor the materials attach to. */
export class Stage {
  readonly renderer: WebGPURenderer;
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(30, 1, 0.1, 120);
  readonly anchor = new Group();
  readonly backdrop: Backdrop;
  private readonly supersample: boolean;

  /**
   * Starts WebGPU and builds the stage. Throws without WebGPU: the renderer would quietly fall back to WebGL,
   * which this app doesn't support.
   */
  static async create(canvas: HTMLCanvasElement, setting: SettingId, supersample: boolean): Promise<Stage> {
    const gpu = (navigator as GpuNavigator).gpu;
    if (!gpu || !(await gpu.requestAdapter())) throw new Error('WebGPU is not available');
    const renderer = new WebGPURenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    await renderer.init();
    if (!(renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend) {
      renderer.dispose();
      throw new Error('WebGPU failed to start');
    }
    return new Stage(renderer, setting, supersample);
  }

  private constructor(renderer: WebGPURenderer, setting: SettingId, supersample: boolean) {
    this.supersample = supersample;
    this.renderer = renderer;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    const pmrem = new PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.85;
    pmrem.dispose();

    this.backdrop = new Backdrop(setting);
    this.scene.add(this.backdrop.sphere, this.backdrop.pool, this.anchor);
  }

  /**
   * Sizes the canvas and places the camera so the object fits in the space left between
   * the top bar (`topInset` px) and the bottom dock (`bottomInset` px).
   */
  fit(width: number, height: number, topInset: number, bottomInset: number): void {
    const dpr = window.devicePixelRatio || 1;
    const budget = Math.sqrt(SUPERSAMPLE_PIXELS / (width * height));
    this.renderer.setPixelRatio(Math.min(2, this.supersample ? Math.max(dpr, budget) : dpr));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    const available = Math.max(height * 0.45, height - topInset - bottomInset);
    const tanHalf = Math.tan(MathUtils.degToRad(this.camera.fov / 2));
    const fraction = Math.min((width / height) * 0.92, available / height);
    const distance = FIT_RADIUS / (tanHalf * fraction);
    this.camera.position.set(0, distance * 0.12, distance);
    this.camera.lookAt(0, 0, 0);
    // shift the view so the object is centred in the free space rather than the whole screen
    this.camera.setViewOffset(width, height, 0, (bottomInset - topInset) / 2, width, height);
    this.camera.updateProjectionMatrix();
  }

  render(dt: number): void {
    this.backdrop.update(dt);
    this.renderer.render(this.scene, this.camera);
  }
}
