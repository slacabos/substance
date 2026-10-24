import { IcosahedronGeometry, Matrix4, MeshBasicMaterial, Vector3 } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { describe, expect, it } from 'vitest';
import { OceanWeather } from '../src/core/oceanWeather';
import { OceanFoamFlecks } from '../src/materials/oceanFoamFlecks';
import { OceanSim } from '../src/materials/oceanSim';

describe('ocean foam flecks', () => {
  it('responds to the live foam strength', () => {
    const foam = new OceanFoamFlecks(true);
    const material = foam.mesh.material as MeshBasicMaterial;
    foam.setIntensity(0);
    expect(material.opacity).toBe(0);
    foam.setIntensity(1.4);
    expect(material.opacity).toBeCloseTo(0.66);
    foam.setIntensity(2.5);
    expect(material.opacity).toBeGreaterThan(0.66);
  });

  it('rides the spherical surface and disappears after its lifetime', () => {
    const source = new IcosahedronGeometry(1, 5);
    source.deleteAttribute('uv');
    const sim = new OceanSim(mergeVertices(source));
    const weather = new OceanWeather();
    const touch = new Vector3(0, 0, 1);
    weather.down(touch);
    const foam = new OceanFoamFlecks(true);
    foam.emit(touch, new Vector3(0, 1, 0), 1, 1);
    sim.update(1 / 60, [{ dir: touch, strength: -0.7 }]);
    foam.update(0.1, sim, weather);
    const transform = new Matrix4();
    foam.mesh.getMatrixAt(0, transform);
    const position = new Vector3().setFromMatrixPosition(transform);
    expect(position.length()).toBeGreaterThan(0.9);
    expect(position.length()).toBeLessThan(1.1);
    for (let i = 0; i < 180; i++) foam.update(1 / 60, sim, weather);
    foam.mesh.getMatrixAt(0, transform);
    expect(transform.determinant()).toBe(0);
  });
});
