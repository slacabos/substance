import type { MaterialId, ShapeId } from '../materials/types';
import { SETTING_ORDER, type SettingId } from '../scene/settings';

/** The parts of the scene the URL can pin down, so a reload or a shared link opens on the same view. */
export interface SceneState {
  material: MaterialId;
  shape: ShapeId;
  setting: SettingId;
  spin: boolean;
}

export const DEFAULT_SCENE: SceneState = { material: 'water', shape: 'orb', setting: 'night', spin: true };

const MATERIALS: readonly MaterialId[] = ['water', 'ocean', 'slime', 'fur'];
const SHAPES: readonly ShapeId[] = ['orb', 'cube'];

/** Reads a scene from a query string such as `?material=fur&shape=cube`. Missing or unknown values keep their default. */
export function parseScene(search: string): SceneState {
  const query = new URLSearchParams(search);
  const pick = <T extends string>(key: string, options: readonly T[], fallback: T): T => {
    const value = query.get(key);
    return options.find((option) => option === value) ?? fallback;
  };
  return {
    material: pick('material', MATERIALS, DEFAULT_SCENE.material),
    shape: pick('shape', SHAPES, DEFAULT_SCENE.shape),
    setting: pick('setting', SETTING_ORDER, DEFAULT_SCENE.setting),
    spin: query.get('spin') === 'off' ? false : DEFAULT_SCENE.spin,
  };
}

/**
 * The query string for a scene, built on top of `base` (other parameters are kept). Defaults are left out so
 * the plain URL stays clean. Returns '' when nothing needs saying.
 */
export function sceneQuery(scene: SceneState, base = ''): string {
  const query = new URLSearchParams(base);
  const put = (key: string, value: string, fallback: string) => {
    if (value === fallback) query.delete(key);
    else query.set(key, value);
  };
  put('material', scene.material, DEFAULT_SCENE.material);
  put('shape', scene.shape, DEFAULT_SCENE.shape);
  put('setting', scene.setting, DEFAULT_SCENE.setting);
  put('spin', scene.spin ? 'on' : 'off', 'on');
  const text = query.toString();
  return text ? `?${text}` : '';
}
