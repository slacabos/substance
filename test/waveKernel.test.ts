import { describe, expect, it } from 'vitest';
import { oceanMesh } from '../src/core/oceanMesh';
import { oceanDetail } from '../src/core/oceanTuning';
import {
  buildWaveKernel,
  fitShellWeights,
  lattice,
  waveSymbol,
  WAVE_MAX_DEPTH,
  WAVE_SLOTS,
} from '../src/core/waveKernel';
import { createDeformGeometry } from '../src/materials/deform';

/** The fitted operator's response to a plane wave on the ideal lattice. */
function latticeSymbol(depth: number, kx: number, ky: number): number {
  const { offsets, shells } = lattice();
  const { weights } = fitShellWeights(depth);
  let sum = 0;
  for (const [x, y] of offsets) {
    const shell = shells.findIndex((s) => Math.abs(s - Math.hypot(x, y)) < 1e-5);
    sum += weights[shell]! * (Math.cos(kx * x + ky * y) - 1);
  }
  return sum;
}

describe('iWave operator', () => {
  it('matches k tanh(kH) on an ideal hex lattice in every direction', () => {
    expect(lattice().offsets).toHaveLength(60);
    for (const depth of [0.25, 1, WAVE_MAX_DEPTH]) {
      for (const k of [0.15, 0.6, 1.0, 1.8]) {
        for (const angle of [0, 0.3, 0.7, 1.1]) {
          const ratio = latticeSymbol(depth, k * Math.cos(angle), k * Math.sin(angle)) / waveSymbol(k, depth);
          // the deepest fit trades some accuracy for staying positive (WAVE_MAX_DEPTH)
          expect(Math.abs(ratio - 1)).toBeLessThan(depth > 2 ? 0.2 : 0.15);
        }
      }
    }
  });

  it('never goes negative, at any wavenumber the lattice can carry, so the sea stays stable', () => {
    for (const depth of [0.1, 0.5, 1, 1.5, WAVE_MAX_DEPTH]) {
      for (let i = 1; i <= 40; i++) {
        for (let j = 0; j <= 12; j++) {
          const k = (4.2 * i) / 40;
          const angle = (j / 12) * (Math.PI / 3);
          expect(latticeSymbol(depth, k * Math.cos(angle), k * Math.sin(angle))).toBeGreaterThan(0);
        }
      }
    }
  });

  it('keeps a flat sea flat and disperses waves on the ocean meshes', () => {
    for (const coarse of [false, true]) {
      const mesh = oceanMesh(createDeformGeometry(true, oceanDetail(coarse)));
      const count = mesh.positions.length / 3;
      const depth = 0.05;
      const kernel = buildWaveKernel(mesh.positions, mesh.neighbors, new Float32Array(count).fill(depth));
      const p = mesh.positions;
      let error = 0;
      let samples = 0;
      for (let i = 3; i < count; i += 211) {
        let sum = 0;
        for (let s = 0; s < WAVE_SLOTS; s++) sum += kernel.weights[i * WAVE_SLOTS + s]!;
        expect(Math.abs(sum)).toBeLessThan(1e-3);
        // a local plane wave cos(k t.x) around vertex i, in its tangent plane
        const n = [p[i * 3]!, p[i * 3 + 1]!, p[i * 3 + 2]!];
        const a = Math.abs(n[1]!) < 0.9 ? [0, 1, 0] : [1, 0, 0];
        const an = a[0]! * n[0]! + a[1]! * n[1]! + a[2]! * n[2]!;
        const t = a.map((v, c) => v - n[c]! * an);
        const tl = Math.hypot(...t);
        for (const ks of [0.15, 0.6, 1.4]) {
          const k = ks / mesh.spacing;
          let response = 0;
          for (let s = 0; s < WAVE_SLOTS; s++) {
            const j = kernel.indices[i * WAVE_SLOTS + s]!;
            const along =
              ((p[j * 3]! - n[0]!) * t[0]! +
                (p[j * 3 + 1]! - n[1]!) * t[1]! +
                (p[j * 3 + 2]! - n[2]!) * t[2]!) /
              tl;
            response += kernel.weights[i * WAVE_SLOTS + s]! * Math.cos(k * along);
          }
          expect(response).toBeGreaterThan(0);
          error += Math.abs(response / waveSymbol(k, depth) - 1);
          samples++;
        }
      }
      // the open sea is three spacings deep on the desktop mesh, at the edge of what the stencil fits
      expect(error / samples).toBeLessThan(0.2);
    }
  });
});
