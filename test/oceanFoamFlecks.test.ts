import { IcosahedronGeometry } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import type { MeshBasicNodeMaterial, WebGPURenderer } from 'three/webgpu';
import { describe, expect, it } from 'vitest';
import { OCEAN_DEFAULTS } from '../src/core/oceanTuning';
import { createDeformMaterial, DeformMode } from '../src/materials/deform';
import { OceanFFT } from '../src/materials/oceanFFT';
import { OceanFoamFlecks } from '../src/materials/oceanFoamFlecks';
import { OceanSim } from '../src/materials/oceanSim';

function flecks() {
  const calls: unknown[] = [];
  const renderer = { compute: (n: unknown) => calls.push(n) } as unknown as WebGPURenderer;
  const source = new IcosahedronGeometry(1, 6);
  source.deleteAttribute('uv');
  const sim = new OceanSim(renderer, mergeVertices(source));
  const waves = new OceanFFT(renderer, true, OCEAN_DEFAULTS);
  const { uniforms } = createDeformMaterial({}, DeformMode.Ocean);
  const foam = new OceanFoamFlecks(renderer, true, {
    buffers: sim.renderBuffers(),
    waves,
    uniforms,
    spacing: sim.spacing,
    count: sim.count,
  });
  return { foam, calls };
}

describe('ocean foam flecks', () => {
  it('responds to the live foam strength', () => {
    const { foam } = flecks();
    const material = foam.mesh.material as MeshBasicNodeMaterial;
    foam.setIntensity(0);
    expect(material.opacity).toBe(0);
    foam.setIntensity(1.4);
    expect(material.opacity).toBeCloseTo(0.66);
    foam.setIntensity(2.5);
    expect(material.opacity).toBeGreaterThan(0.66);
  });

  it('starts and moves flecks in one GPU dispatch a frame, with nothing from the CPU', () => {
    const { foam, calls } = flecks();
    foam.update(1 / 60);
    foam.update(1 / 60);
    expect(calls).toHaveLength(2);
    // emission over the surface's vertices, then the step over the flecks
    expect(calls[0]).toHaveLength(2);
  });
});
