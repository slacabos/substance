import { BufferGeometry, Color, Float32BufferAttribute, Mesh, Vector3 } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { SHORE_SITES, shoreElevation, shoreNoise, shoreRadius, shoreSiteDistance } from '../core/oceanShores';
import { shoreSurface } from '../nodes/shore';
import type { SurfInputs } from '../nodes/surf';
import { blendedAt, seaHeightAt } from '../nodes/oceanSurface';
import {
  attribute,
  float,
  Fn,
  normalize,
  normalLocal,
  positionGeometry,
  positionLocal,
  varyingProperty,
  vec3,
} from 'three/tsl';
import type { StorageBufferNode } from 'three/webgpu';

const SAND = new Color('#c4ab80');
const WET_SAND = new Color('#8f7d5b');
const WARM_SAND = new Color('#c99d6a');
const PALE_SAND = new Color('#ddd0b0');
const SHELF = new Color('#c9bf96');
const DEEP = new Color('#2c4a52');
const color = new Color();
/** The sim's buffers the land reads the sea from (OceanSim.renderBuffers). */
type SeaBuffers = Parameters<typeof seaHeightAt>[1] & { slope: StorageBufferNode<'vec4'> };
const UP = new Vector3(0, 1, 0);
const SIDE = new Vector3(1, 0, 0);

const smooth = (a: number, b: number, value: number): number => {
  const t = Math.min(1, Math.max(0, (value - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Continuous raised terrain with shared vertices, a narrow wet bank and a submerged foundation. */
export class OceanShores {
  readonly mesh: Mesh;
  private readonly directions: Float32Array;
  private readonly elevations: Float32Array;
  private morph = Number.NaN;

  constructor(
    coarse: boolean,
    surf: SurfInputs,
    sea: SeaBuffers,
    sand?: { bed: StorageBufferNode<'vec4'>; slope: StorageBufferNode<'vec4'> },
  ) {
    const skirtRings = 8;
    const detail = coarse ? 0.5 : 1;
    const positions: number[] = [];
    const directions: number[] = [];
    const elevations: number[] = [];
    const indices: number[] = [];
    const siteOf: number[] = [];
    const dir = new Vector3();
    for (const [k, site] of SHORE_SITES.entries()) {
      // as fine on the continent's long coast as on a small island's
      const segments = Math.round(Math.min(360, Math.max(96, site.radius * 230)) * detail);
      const landRings = Math.round(Math.min(140, Math.max(36, site.radius * 90)) * detail);
      const rings = landRings + skirtRings;
      const center = positions.length / 3;
      const addVertex = () => {
        const elevation = shoreElevation(site, dir);
        directions.push(dir.x, dir.y, dir.z);
        siteOf.push(k);
        elevations.push(elevation);
        positions.push(dir.x * (1 + elevation), dir.y * (1 + elevation), dir.z * (1 + elevation));
      };
      dir.copy(site.center);
      addVertex();
      // each ring holds as many vertices as its circumference needs, so the middle isn't a fan of slivers
      const starts: number[] = [];
      const counts: number[] = [];
      const spacing = (2 * Math.PI * Math.sin(site.radius)) / segments;
      for (let ring = 1; ring <= rings; ring++) {
        // rings twice as close at the coast as in the middle, where the beach needs them; then an
        // underwater skirt a fixed width out
        const t = Math.min(1, ring / landRings);
        const inner = (t + Math.sin((t * Math.PI) / 2)) / 2;
        const outer = Math.max(0, ring - landRings) / skirtRings;
        const reach = site.radius * inner + Math.min(site.radius * 0.4, 0.2) * outer;
        const count =
          ring >= landRings
            ? segments
            : Math.max(8, Math.min(segments, Math.round((2 * Math.PI * Math.sin(reach)) / spacing)));
        starts.push(positions.length / 3);
        counts.push(count);
        for (let sector = 0; sector < count; sector++) {
          const angle = (sector / count) * Math.PI * 2;
          const coast = shoreRadius(site, angle);
          const arc = coast * inner + Math.min(coast * 0.4, 0.2) * outer;
          dir
            .copy(site.center)
            .multiplyScalar(Math.cos(arc))
            .addScaledVector(site.east, Math.cos(angle) * Math.sin(arc))
            .addScaledVector(site.north, Math.sin(angle) * Math.sin(arc))
            .normalize();
          addVertex();
        }
      }
      for (let sector = 0; sector < counts[0]!; sector++) {
        indices.push(center, starts[0]! + sector, starts[0]! + ((sector + 1) % counts[0]!));
      }
      // stitch each ring to the next, advancing whichever side's next vertex comes first round the circle
      for (let ring = 0; ring < rings - 1; ring++) {
        const [a, b] = [counts[ring]!, counts[ring + 1]!];
        const [inner, outer] = [starts[ring]!, starts[ring + 1]!];
        let i = 0;
        let j = 0;
        while (i < a || j < b) {
          if (j >= b || (i < a && (i + 1) / a <= (j + 1) / b)) {
            indices.push(inner + (i % a), outer + (j % b), inner + ((i + 1) % a));
            i++;
          } else {
            indices.push(outer + (j % b), outer + ((j + 1) % b), inner + (i % a));
            j++;
          }
        }
      }
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    this.directions = new Float32Array(directions);
    this.elevations = new Float32Array(elevations);
    const colors = new Float32Array(positions.length);
    // how wet each vertex's sand is, for the close-up detail in nodes/shore.ts
    const terrain = new Float32Array(elevations.length);
    for (let i = 0; i < elevations.length; i++) {
      dir.fromArray(this.directions, i * 3);
      const height = elevations[i]!;
      const grain = shoreNoise(dir.x * 173, dir.y * 173, dir.z * 173);
      const wet = 1 - smooth(0.0045, 0.009, height);
      terrain[i] = wet;
      // broad drifts of warmer and paler sand, and dune troughs a little darker than their crests
      const drift = shoreNoise(dir.x * 6, dir.y * 6 + 3.1, dir.z * 6);
      const trough =
        1 - Math.abs(shoreNoise(dir.x * 26 + SHORE_SITES[siteOf[i]!]!.seed, dir.y * 26, dir.z * 26));
      color
        .copy(SAND)
        .lerp(WARM_SAND, Math.max(0, drift) * 0.7)
        .lerp(PALE_SAND, Math.max(0, -drift) * 0.6)
        .multiplyScalar(0.86 + 0.14 * trough)
        .lerp(WET_SAND, wet)
        // below the waterline the shelf is pale sand, darkening with depth, which is what makes clear shallow
        // water turquoise
        .lerp(SHELF, 1 - smooth(0.001, 0.004, height))
        .lerp(DEEP, 1 - smooth(-0.06, -0.01, height))
        .multiplyScalar(0.92 + grain * 0.06);
      colors.set([color.r, color.g, color.b], i * 3);
    }
    geometry.setAttribute('terrain', new Float32BufferAttribute(terrain, 1));
    // for the swash: the way out to sea and how far the shoreline is (as one offset), and the site
    const swash = new Float32Array(elevations.length * 4);
    const t1 = new Vector3();
    const t2 = new Vector3();
    const probe = new Vector3();
    for (let i = 0; i < elevations.length; i++) {
      dir.fromArray(this.directions, i * 3);
      const site = SHORE_SITES[siteOf[i]!]!;
      t1.crossVectors(dir, Math.abs(dir.y) < 0.9 ? UP : SIDE).normalize();
      t2.crossVectors(dir, t1);
      const slope = (t: Vector3) =>
        (shoreSiteDistance(site, probe.copy(dir).addScaledVector(t, 1e-3).normalize()) -
          shoreSiteDistance(site, probe.copy(dir).addScaledVector(t, -1e-3).normalize())) /
        2e-3;
      const seaward = t1.clone().multiplyScalar(slope(t1)).addScaledVector(t2, slope(t2)).normalize();
      const ashore = Math.max(0, -shoreSiteDistance(site, dir));
      swash.set([seaward.x * ashore, seaward.y * ashore, seaward.z * ashore, siteOf[i]!], i * 4);
    }
    geometry.setAttribute('swash', new Float32BufferAttribute(swash, 4));
    geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
    const material = new MeshStandardNodeMaterial();
    // the sea's own height at each vertex's foot on the shoreline, so the hand's waves run up the beach
    const seaAtFoot = varyingProperty('float', 'vSeaAtFoot');
    // and how far the sand has been dug or piled here (the sim's sand bed, blended onto this finer mesh)
    const dug = varyingProperty('float', 'vDug');
    material.positionNode = Fn(() => {
      const swashFoot = attribute('swash', 'vec4');
      const p = normalize(positionGeometry);
      seaAtFoot.assign(seaHeightAt(normalize(p.add(swashFoot.xyz)), sea));
      const change = sand ? blendedAt(p, sea, (i) => sand.bed.element(i).x) : float(0);
      dug.assign(change);
      if (sand) {
        // the dug relief's slope, blended like the height, tilts the land's normal (a bump from the interpolated
        // height would be flat per triangle, faceted)
        const slope = vec3(
          blendedAt(p, sea, (i) => sand.slope.element(i).x),
          blendedAt(p, sea, (i) => sand.slope.element(i).y),
          blendedAt(p, sea, (i) => sand.slope.element(i).z),
        );
        normalLocal.assign(normalize(normalLocal.sub(slope)));
      }
      return positionLocal.add(p.mul(change));
    })();
    const surface = shoreSurface(surf, seaAtFoot, dug);
    material.colorNode = surface.color;
    material.roughnessNode = surface.roughness;
    material.normalNode = surface.normal;
    this.mesh = new Mesh(geometry, material);
    this.mesh.frustumCulled = false;
    this.updateMorph(0);
  }

  updateMorph(morph: number): void {
    if (Math.abs(morph - this.morph) < 1e-4) return;
    this.morph = morph;
    const position = this.mesh.geometry.getAttribute('position');
    for (let i = 0; i < this.elevations.length; i++) {
      const x = this.directions[i * 3]!;
      const y = this.directions[i * 3 + 1]!;
      const z = this.directions[i * 3 + 2]!;
      const power = Math.abs(x) ** 10 + Math.abs(y) ** 10 + Math.abs(z) ** 10;
      const radius = 1 + (0.8 / power ** 0.1 - 1) * morph + this.elevations[i]!;
      position.setXYZ(i, x * radius, y * radius, z * radius);
    }
    position.needsUpdate = true;
    this.mesh.geometry.computeVertexNormals();
  }
}
