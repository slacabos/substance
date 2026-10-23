import {
  ACESFilmicToneMapping,
  Group,
  MathUtils,
  PerspectiveCamera,
  PMREMGenerator,
  Scene,
  WebGLRenderer,
} from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { FIT_RADIUS } from '../config';
import { createBackdrop, createLightPool } from './backdrop';

/** Renderer, camera, lighting environment and the spinning anchor the materials attach to. */
export class Stage {
  readonly renderer: WebGLRenderer;
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(30, 1, 0.1, 120);
  readonly anchor = new Group();
  readonly pool = createLightPool();

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    const pmrem = new PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.85;
    pmrem.dispose();

    this.scene.add(createBackdrop(), this.pool.mesh, this.anchor);
  }

  /**
   * Sizes the canvas and places the camera so the object fits in the space left between
   * the top bar (`topInset` px) and the bottom dock (`bottomInset` px).
   */
  fit(width: number, height: number, topInset: number, bottomInset: number): void {
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

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }
}
