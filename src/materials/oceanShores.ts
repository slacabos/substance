import { BufferGeometry, Color, Float32BufferAttribute, Mesh, MeshStandardMaterial, Vector3 } from 'three';
import { SHORE_SITES, shoreElevation, shoreRadius, shoreNoise } from '../core/oceanShores';

const SAND = new Color('#9d8967');
const STONE = new Color('#535a58');
const GRASS = new Color('#3d5e40');
const WET = new Color('#344644');
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
    const dir = new Vector3();
    for (const site of SHORE_SITES) {
      const center = positions.length / 3;
      const addVertex = () => {
        const elevation = shoreElevation(site, dir);
        directions.push(dir.x, dir.y, dir.z);
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
    const normals = geometry.getAttribute('normal');
    for (let i = 0; i < elevations.length; i++) {
      dir.fromArray(this.directions, i * 3);
      const slope = 1 - (normals.getX(i) * dir.x + normals.getY(i) * dir.y + normals.getZ(i) * dir.z);
      const height = elevations[i]!;
      const grain = shoreNoise(dir.x * 173, dir.y * 173, dir.z * 173);
      color
        .copy(STONE)
        .lerp(GRASS, smooth(0.06, 0.12, height) * (1 - smooth(0.08, 0.32, slope)) * 0.9)
        .lerp(SAND, (1 - smooth(0.018, 0.06, height)) * (1 - smooth(0.12, 0.5, slope)) * 0.65)
        // a narrow wet band at the waterline; below it the shelf is pale sand, darkening with depth, which is
        // what makes clear shallow water turquoise
        .lerp(WET, (1 - smooth(0.012, 0.024, height)) * smooth(0.002, 0.01, height))
        .lerp(SHELF, 1 - smooth(0.002, 0.01, height))
        .lerp(DEEP, 1 - smooth(-0.06, -0.01, height))
        .multiplyScalar(0.85 + grain * 0.13);
      colors.set([color.r, color.g, color.b], i * 3);
    }
    geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
    this.mesh = new Mesh(geometry, new MeshStandardMaterial({ vertexColors: true, roughness: 0.93 }));
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
