import { IcosahedronGeometry, Vector3 } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import type { WebGPURenderer } from 'three/webgpu';
import { describe, expect, it } from 'vitest';
import { OCEAN_DEPTH } from '../src/core/oceanFFT';
import { createOceanField } from '../src/core/oceanField';
import { SHELF_EDGE, shoreDistance } from '../src/core/oceanShores';
import { OceanSim, oceanPressLoad, oceanReleaseImpulse } from '../src/materials/oceanSim';

/** Records compute dispatches instead of running them; no GPU in tests. */
function stubRenderer() {
  const calls: unknown[][] = [];
  const renderer = {
    compute: (nodes: unknown) => calls.push(Array.isArray(nodes) ? nodes : [nodes]),
  } as unknown as WebGPURenderer;
  return { renderer, calls };
}

const sphere = (detail: number) => {
  const source = new IcosahedronGeometry(1, detail);
  source.deleteAttribute('uv');
  return mergeVertices(source);
};

describe('ocean surface simulation', () => {
  it('substeps longer frames and finishes once per frame', () => {
    const { renderer, calls } = stubRenderer();
    const sim = new OceanSim(renderer, sphere(8));
    sim.update(1 / 60, []);
    // one step (momentum, height), then splash landings, the volume correction (clear, sum, correct), surface
    // and slope
    expect(calls.at(-1)).toHaveLength(2 + 6);
    sim.update(1 / 20, [{ dir: new Vector3(0, 0, 1), strength: -0.55 }]);
    expect(calls.at(-1)).toHaveLength(3 * 2 + 6);
    sim.update(0.5, []);
    // A changed substep duration first rescales the height history.
    expect(calls.at(-1)).toHaveLength(1 + 4 * 2 + 6);
    sim.update(0, []);
    expect(calls).toHaveLength(3);
  });

  it('rescales history once when the frame changes from one 24 ms step to two 13 ms steps', () => {
    const { renderer, calls } = stubRenderer();
    const sim = new OceanSim(renderer, sphere(8));
    sim.update(0.024, []);
    expect(calls.at(-1)).toHaveLength(1 + 2 + 6);
    expect(calls.at(-1)![0]).toBe(sim['rescaleNodes'][0]);
    expect(sim['uniforms'].stepScale.value).toBeCloseTo(0.024 / (1 / 60));

    sim.update(0.026, []);
    expect(calls.at(-1)).toHaveLength(1 + 2 * 2 + 6);
    expect(calls.at(-1)![0]).toBe(sim['rescaleNodes'][1]);
    expect(sim['uniforms'].stepScale.value).toBeCloseTo(0.013 / 0.024);

    sim.update(0, []);
    sim.update(0.026, []);
    expect(calls.at(-1)).toHaveLength(2 * 2 + 6);
    expect(sim['uniforms'].stepScale.value).toBe(1);

    sim.update(0.024, []);
    expect(calls.at(-1)).toHaveLength(1 + 2 + 6);
    expect(sim['uniforms'].stepScale.value).toBeCloseTo(0.024 / 0.013);
  });

  it('takes presses and resets without reading anything back', () => {
    const { renderer, calls } = stubRenderer();
    const sim = new OceanSim(renderer, sphere(8));
    sim.setPressure(new Vector3(0, 1, 0), 0.4);
    sim.update(1 / 60, []);
    sim.setPressure(null, 0);
    sim.reset();
    expect(calls.length).toBe(2);
    // two landing slots per vertex, for the splash fluid
    expect(sim.landings.value.count).toBe(sim.count * 2);
  });

  it('loads harder the longer a press is held, and releases a bigger wave', () => {
    expect(oceanPressLoad(0)).toBe(0);
    expect(oceanPressLoad(0.8)).toBeGreaterThan(oceanPressLoad(0.1));
    expect(oceanReleaseImpulse(0.8)).toBeGreaterThan(oceanReleaseImpulse(0.08) * 1.5);
  });
});

describe('coastal ocean field', () => {
  it('stands its land above the sea, shoals the sea toward the bank, and keeps every triangle', () => {
    const geometry = sphere(18);
    const fullIndexCount = geometry.getIndex()!.count;
    const field = createOceanField(geometry, true);
    // the water may spread anywhere the sand is dug below it
    expect(geometry.getIndex()!.count).toBe(fullIndexCount);
    const dir = new Vector3();
    let dry = 0;
    let shallow = 0;
    for (let i = 0; i < field.count; i++) {
      const distance = shoreDistance(dir.fromArray(field.positions, i * 3));
      expect(field.wet[i]).toBe(distance > 0 ? 1 : 0);
      if (!field.wet[i]) {
        dry++;
        // land is ground above the sea: negative depth
        expect(field.depth[i]).toBeLessThan(0);
      }
      if (field.wet[i] && distance < 0.01) {
        shallow++;
        expect(field.depth[i]).toBeLessThan(OCEAN_DEPTH);
      }
      if (distance > SHELF_EDGE) expect(field.depth[i]).toBeCloseTo(OCEAN_DEPTH, 6);
    }
    // about half the orb is land
    expect(dry).toBeGreaterThan(field.count * 0.4);
    expect(shallow).toBeGreaterThan(0);
  });

  it('keeps the coast within WebGPU`s eight vertex buffers', () => {
    const geometry = sphere(12);
    new OceanSim(stubRenderer().renderer, geometry, true);
    expect(Object.keys(geometry.attributes).length).toBeLessThanOrEqual(8);
    expect(geometry.getAttribute('oceanShoreData').itemSize).toBe(4);
  });
});
