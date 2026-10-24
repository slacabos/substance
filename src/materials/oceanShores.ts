import { BufferGeometry, Color, Float32BufferAttribute, Mesh, Vector3 } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { SHORE_SITES, shoreElevation, shoreNoise, shoreRadius, shoreSiteDistance } from '../core/oceanShores';
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
    const segments = coarse ? 64 : 96;
    const landRings = coarse ? 24 : 36;
    const skirtRings = 8;
    const rings = landRings + skirtRings;
    const positions: number[] = [];
    const directions: number[] = [];
    const elevations: number[] = [];
    const indices: number[] = [];
    const siteOf: number[] = [];
    const dir = new Vector3();
    for (const [k, site] of SHORE_SITES.entries()) {
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
      for (let ring = 1; ring <= rings; ring++) {
        const radius = ring <= landRings ? ring / landRings : 1 + ((ring - landRings) / skirtRings) * 0.4;
        for (let sector = 0; sector < segments; sector++) {
          const angle = (sector / segments) * Math.PI * 2;
          const arc = shoreRadius(site, angle) * radius;
          dir
            .copy(site.center)
            .multiplyScalar(Math.cos(arc))
            .addScaledVector(site.east, Math.cos(angle) * Math.sin(arc))
            .addScaledVector(site.north, Math.sin(angle) * Math.sin(arc))
            .normalize();
          addVertex();
        }
      }
      for (let sector = 0; sector < segments; sector++) {
        indices.push(center, center + 1 + sector, center + 1 + ((sector + 1) % segments));
      }
      for (let ring = 0; ring < rings - 1; ring++) {
        for (let sector = 0; sector < segments; sector++) {
          const next = (sector + 1) % segments;
          const a = center + 1 + ring * segments + sector;
          const b = a + segments;
          const c = center + 1 + ring * segments + next;
          indices.push(a, b, c, b, c + segments, c);
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
      // the beach, and low flat ground; then grass on gentle slopes high enough above it, rock elsewhere
      const sand = Math.max(
        1 - smooth(site.beach * 0.75, site.beach * 1.15 + 0.006, ashore),
        (1 - smooth(0.02, 0.035, height)) * (1 - smooth(0.12, 0.45, slope)),
      );
      // grass from a little above the beach, giving way to bare rock on the highest ground and steep slopes
      const above = (height - 0.016) / site.height;
      const grass =
        (1 - sand) *
        smooth(0.2, 0.5, above) *
        (1 - smooth(0.75, 0.95, above)) *
        (1 - smooth(0.08, 0.3, slope));
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
