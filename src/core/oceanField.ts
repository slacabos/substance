import { Vector3, type BufferGeometry } from 'three';
import { OCEAN_DEPTH } from './oceanFFT';
import { buildDirectionLookup } from './oceanLookup';
import { oceanMesh } from './oceanMesh';
import {
  shelfLag,
  SHORE_SITES,
  SHORE_WATERLINE,
  shoreDistance,
  shoreGroundHeight,
  shoreSiteDistance,
} from './oceanShores';
import { vertexAreas } from './shallowWater';
import { buildWaveKernel, type WaveKernel } from './waveKernel';

/** Everything about the ocean surface that never changes: mesh, coast, depth, wave operator, lookup. */
export interface OceanField {
  count: number;
  spacing: number;
  positions: Float32Array;
  neighbors: Uint32Array;
  /** 1 where the sea starts, 0 on land (which wets as water reaches it). */
  wet: Uint8Array;
  /** Angular distance from the shore (negative on land); 1 everywhere on the open ocean. */
  shoreDistance: Float32Array;
  /** Unit tangent pointing away from the nearest shore, three per vertex (zero on the open ocean). */
  shoreNormal: Float32Array;
  /** Which island (SHORE_SITES) is nearest, whose swell the surf here follows. */
  shoreSite: Float32Array;
  /** How far behind a crest falls crossing the shelf to here (shelfLag, core/oceanShores.ts). */
  shelfLag: Float32Array;
  /** Still-water depth, negative where the ground stands above the sea. */
  depth: Float32Array;
  areas: Float32Array;
  kernel: WaveKernel;
  /** Nearest-vertex cube map (core/oceanLookup.ts). */
  lookup: Uint32Array;
  /** Per-vertex noise: how strongly a spot answers a push, and how readily it foams. */
  forceGrain: Float32Array;
  foamGrain: Float32Array;
}

const fract = (x: number): number => x - Math.floor(x);

/**
 * Builds the field for an indexed icosphere. On a coast, land vertices become walls and the triangles wholly
 * on land are dropped from the geometry, so the water mesh ends at the shore.
 */
export function createOceanField(geometry: BufferGeometry, coastal: boolean): OceanField {
  const mesh = oceanMesh(geometry);
  const { positions, neighbors } = mesh;
  const count = positions.length / 3;
  const distance = new Float32Array(count).fill(1);
  const normal = new Float32Array(count * 3);
  const site = new Float32Array(count);
  const lag = new Float32Array(count);
  const depth = new Float32Array(count).fill(OCEAN_DEPTH);
  const wet = new Uint8Array(count).fill(1);
  const dir = new Vector3();

  if (coastal) {
    for (let i = 0; i < count; i++) distance[i] = shoreDistance(dir.fromArray(positions, i * 3));
    for (let i = 0; i < count; i++) {
      const [x, y, z] = [positions[i * 3]!, positions[i * 3 + 1]!, positions[i * 3 + 2]!];
      // the shore distance's gradient over the neighbours points out to sea
      let gx = 0;
      let gy = 0;
      let gz = 0;
      for (let k = 0; k < 6; k++) {
        const j = neighbors[i * 6 + k]!;
        const projection = x * positions[j * 3]! + y * positions[j * 3 + 1]! + z * positions[j * 3 + 2]!;
        const tx = positions[j * 3]! - x * projection;
        const ty = positions[j * 3 + 1]! - y * projection;
        const tz = positions[j * 3 + 2]! - z * projection;
        const factor = (distance[j]! - distance[i]!) / Math.max(1e-6, tx * tx + ty * ty + tz * tz);
        gx += tx * factor;
        gy += ty * factor;
        gz += tz * factor;
      }
      const length = Math.hypot(gx, gy, gz) || 1;
      normal.set([gx / length, gy / length, gz / length], i * 3);
      dir.fromArray(positions, i * 3);
      let nearest = Infinity;
      SHORE_SITES.forEach((s, k) => {
        const d = shoreSiteDistance(s, dir);
        if (d < nearest) [nearest, site[i]] = [d, k];
      });
      lag[i] = shelfLag(distance[i]!);
      wet[i] = distance[i]! > 0 ? 1 : 0;
      // The seabed rises toward the bank and the open sea keeps the ocean's depth; land stands above the sea
      // (negative depth), dry until water reaches it. Every triangle stays, so water can spread wherever the
      // ground is dug below it.
      const ground = shoreGroundHeight(dir.fromArray(positions, i * 3));
      depth[i] = wet[i]
        ? Math.min(OCEAN_DEPTH, Math.max(0.002, -ground))
        : Math.min(-SHORE_WATERLINE, -ground);
    }
  }

  const forceGrain = new Float32Array(count);
  const foamGrain = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const [x, y, z] = [positions[i * 3]!, positions[i * 3 + 1]!, positions[i * 3 + 2]!];
    forceGrain[i] = 0.85 + Math.sin((x * 37 + y * 51 + z * 83) * 11) * 0.15;
    foamGrain[i] = fract(Math.sin((x * 43.7 + y * 91.3 + z * 67.9) * 67.3) * 43758.5453);
  }

  return {
    count,
    spacing: mesh.spacing,
    positions,
    neighbors,
    wet,
    shoreDistance: distance,
    shoreNormal: normal,
    shoreSite: site,
    shelfLag: lag,
    depth,
    areas: vertexAreas(mesh),
    // land holds no waves of its own; its operator is the shallowest water's
    kernel: buildWaveKernel(
      positions,
      neighbors,
      depth.map((d) => Math.max(d, 0.002)),
    ),
    lookup: buildDirectionLookup(positions),
    forceGrain,
    foamGrain,
  };
}
