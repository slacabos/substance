import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { Scene } from 'three';
import { Stage } from '../src/scene/stage';

/** Exercise resolution lifecycle without allocating a renderer or requesting a GPU. */
function stageStub() {
  const stage = Object.create(Stage.prototype) as Stage;
  const renderer = { setPixelRatio: vi.fn(), setSize: vi.fn() };
  Object.assign(stage, {
    renderer,
    scene: new Scene(),
    detailMode: 0,
    coarse: true,
    supersample: false,
    plane: 0,
    planeResolution: false,
    screen: { width: 390, height: 844, shift: 0 },
    backdrop: { setPlane: vi.fn() },
    place: vi.fn(),
  });
  return { stage, renderer };
}

describe('phone plane resolution lifecycle', () => {
  beforeEach(() => vi.stubGlobal('window', { devicePixelRatio: 3 }));
  afterEach(() => vi.unstubAllGlobals());

  it('caps once on entry, keeps the cap during reversal and restores on reaching the orb', () => {
    const { stage, renderer } = stageStub();
    stage.preparePlane();
    expect(renderer.setPixelRatio).toHaveBeenLastCalledWith(1.5);
    stage.setPlane(0.2);
    stage.setPlane(0.8);
    stage.setPlane(0.1);
    expect(renderer.setSize).toHaveBeenCalledTimes(1);
    stage.setPlane(0);
    expect(renderer.setPixelRatio).toHaveBeenLastCalledWith(2);
    expect(renderer.setSize).toHaveBeenCalledTimes(2);
    stage.setPlane(0);
    expect(renderer.setSize).toHaveBeenCalledTimes(2);
  });

  it('restores the budget when entry is cancelled before the first animated frame', () => {
    const { stage, renderer } = stageStub();
    stage.preparePlane();
    stage.setPlane(0);
    expect(renderer.setPixelRatio).toHaveBeenLastCalledWith(2);
  });
});
