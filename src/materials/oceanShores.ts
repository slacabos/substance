import { BufferGeometry, Color, Float32BufferAttribute, Mesh, Vector3 } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
  beachWidth,
  SHORE_SITES,
  shoreElevation,
  shoreNoise,
  shoreRadius,
  shoreSiteDistance,
} from '../core/oceanShores';
import { shoreSurface } from '../nodes/shore';

const SAND = new Color('#c9b48a');
const WET_SAND = new Color('#8f7d5b');
const STONE = new Color('#535a58');
const GRASS = new Color('#3d5e40');
const SHELF = new Color('#c9bf96');
const DEEP = new Color('#2c4a52');
const color = new Color();

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

  constructor(coarse: boolean) {
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
    // how much of each surface a vertex is (rock, sand, grass) and how wet its sand is, for the close-up
    // detail in nodes/shore.ts
    const terrain = new Float32Array(elevations.length * 4);
    const normals = geometry.getAttribute('normal');
    for (let i = 0; i < elevations.length; i++) {
      dir.fromArray(this.directions, i * 3);
      const site = SHORE_SITES[siteOf[i]!]!;
      const slope = 1 - (normals.getX(i) * dir.x + normals.getY(i) * dir.y + normals.getZ(i) * dir.z);
      const height = elevations[i]!;
      const ashore = -shoreSiteDistance(site, dir);
      const grain = shoreNoise(dir.x * 173, dir.y * 173, dir.z * 173);
      // The beach and low flat ground are sand. Inland it's sandy too, with patches of grass where the ground
      // has risen a little, giving way to bare rock on the highest ground and steep slopes.
      const beach = Math.max(
        1 - smooth(beachWidth(site, dir) * 0.75, beachWidth(site, dir) * 1.15 + 0.006, ashore),
        (1 - smooth(0.02, 0.035, height)) * (1 - smooth(0.12, 0.45, slope)),
      );
      const above = (height - 0.016) / site.height;
      const patch =
        shoreNoise(dir.x * 9 + site.seed, dir.y * 9, dir.z * 9) +
        0.35 * shoreNoise(dir.x * 23, dir.y * 23 + site.seed, dir.z * 23);
      const grassy = smooth(-0.3, 0.45, patch) * smooth(0.05, 0.3, above) * (1 - smooth(0.08, 0.3, slope));
      const bare = smooth(0.55, 0.85, above) + smooth(0.15, 0.4, slope);
      const grass = (1 - beach) * grassy * (1 - smooth(0.75, 0.95, above));
      const sand = beach + (1 - beach) * (1 - grassy) * Math.max(0, 1 - bare);
      const wet = sand * (1 - smooth(0.0045, 0.009, height));
      terrain.set([Math.max(0, 1 - sand - grass), sand, grass, wet], i * 4);
      color
        .copy(STONE)
        .lerp(GRASS, grass / Math.max(1e-3, 1 - sand))
        .lerp(SAND, sand)
        .lerp(WET_SAND, wet)
        // below the waterline the shelf is pale sand, darkening with depth, which is what makes clear shallow
        // water turquoise
        .lerp(SHELF, 1 - smooth(0.001, 0.004, height))
        .lerp(DEEP, 1 - smooth(-0.06, -0.01, height))
        .multiplyScalar(0.9 + grain * 0.08);
      colors.set([color.r, color.g, color.b], i * 3);
    }
    geometry.setAttribute('terrain', new Float32BufferAttribute(terrain, 4));
    geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
    const material = new MeshStandardNodeMaterial({ vertexColors: true });
    const surface = shoreSurface();
    material.colorNode = surface.tone;
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
