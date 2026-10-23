import { describe, expect, it } from 'vitest';
import { DEFAULT_SCENE, parseScene, sceneQuery } from '../src/core/scene';

describe('scene query strings', () => {
  it('falls back to the default scene', () => {
    expect(parseScene('')).toEqual(DEFAULT_SCENE);
  });

  it('reads every part of the scene', () => {
    expect(parseScene('?material=fur&shape=cube&setting=studio&spin=off')).toEqual({
      material: 'fur',
      shape: 'cube',
      setting: 'studio',
      spin: false,
    });
  });

  it('ignores unknown values', () => {
    expect(parseScene('?material=lava&shape=cone&setting=noon&spin=maybe')).toEqual(DEFAULT_SCENE);
  });

  it('leaves defaults out of the URL', () => {
    expect(sceneQuery(DEFAULT_SCENE)).toBe('');
    expect(sceneQuery({ ...DEFAULT_SCENE, material: 'slime' })).toBe('?material=slime');
  });

  it('round-trips a scene and keeps unrelated parameters', () => {
    const scene = { material: 'ocean', shape: 'cube', setting: 'dusk', spin: false } as const;
    const query = sceneQuery(scene, '?debug=1&material=fur');
    expect(query).toBe('?debug=1&material=ocean&shape=cube&setting=dusk&spin=off');
    expect(parseScene(query)).toEqual(scene);
  });
});
