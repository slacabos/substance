import { BoxGeometry, Mesh, MeshBasicMaterial, Scene } from 'three';
import { describe, expect, it } from 'vitest';
import { setSceneDetailMode } from '../src/nodes/detail';

describe('detail shader variants', () => {
  it('invalidates only at variant boundaries and gives each endpoint its own cache key', () => {
    const scene = new Scene();
    const material = new MeshBasicMaterial();
    scene.add(new Mesh(new BoxGeometry(), material));
    const keys = [];
    for (const mode of [0, 0.5, 1]) {
      const before = material.version;
      setSceneDetailMode(scene, mode);
      expect(material.version).toBe(before + 1);
      keys.push(material.customProgramCacheKey());
      setSceneDetailMode(scene, mode);
      expect(material.version).toBe(before + 1);
    }
    expect(new Set(keys).size).toBe(3);
    setSceneDetailMode(scene, 0);
    expect(material.customProgramCacheKey()).toBe(keys[0]);
  });
});
