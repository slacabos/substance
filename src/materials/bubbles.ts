import { InstancedMesh, MeshStandardMaterial, Object3D, SphereGeometry, Vector3 } from 'three';

// Mirror-like spheres read as air bubbles once seen through a refractive material.
const geometry = new SphereGeometry(1, 16, 12);
const material = new MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.06 });
const dummy = new Object3D();

interface RisingBubble {
  x: number;
  y: number;
  z: number;
  r: number;
  v: number;
  phase: number;
}

/** Bubbles that rise in world space inside a radius-0.8 ball, so they stay "up" while the object spins. */
export class RisingBubbles {
  readonly mesh: InstancedMesh;
  private readonly bubbles: RisingBubble[];

  constructor(count: number) {
    this.mesh = new InstancedMesh(geometry, material, count);
    this.mesh.frustumCulled = false;
    this.bubbles = Array.from({ length: count }, () =>
      this.respawn({ x: 0, y: 0, z: 0, r: 0, v: 0, phase: 0 }, true),
    );
  }

  private respawn(b: RisingBubble, anywhere: boolean): RisingBubble {
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.sqrt(Math.random()) * 0.55;
    b.x = Math.cos(angle) * radius;
    b.z = Math.sin(angle) * radius;
    const floor = -Math.sqrt(Math.max(0, 0.64 - b.x * b.x - b.z * b.z));
    b.y = anywhere ? floor + Math.random() * -2 * floor : floor;
    b.r = 0.007 + Math.random() * Math.random() * 0.026;
    b.v = 0.08 + b.r * 6 + Math.random() * 0.05;
    b.phase = Math.random() * Math.PI * 2;
    return b;
  }

  update(time: number, dt: number): void {
    this.bubbles.forEach((b, i) => {
      b.y += b.v * dt;
      if (b.x * b.x + b.y * b.y + b.z * b.z > 0.62) this.respawn(b, false);
      dummy.position.set(
        b.x + Math.sin(time * 3 + b.phase) * 0.01,
        b.y,
        b.z + Math.cos(time * 2.6 + b.phase) * 0.01,
      );
      dummy.scale.setScalar(b.r);
      dummy.updateMatrix();
      this.mesh.setMatrixAt(i, dummy.matrix);
    });
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

/** Bubbles trapped in place in object space, drifting slightly and squashing with the material. */
export class TrappedBubbles {
  readonly mesh: InstancedMesh;
  private readonly bubbles: { p: Vector3; r: number; phase: number }[];

  constructor(count: number) {
    this.mesh = new InstancedMesh(geometry, material, count);
    this.mesh.frustumCulled = false;
    this.bubbles = Array.from({ length: count }, () => ({
      p: new Vector3().randomDirection().multiplyScalar(0.15 + Math.random() * 0.45),
      r: 0.015 + Math.random() * Math.random() * 0.05,
      phase: Math.random() * Math.PI * 2,
    }));
  }

  update(time: number, squish: number): void {
    this.bubbles.forEach((b, i) => {
      dummy.position.copy(b.p).multiplyScalar(squish);
      dummy.position.y += Math.sin(time * 0.5 + b.phase) * 0.015;
      dummy.scale.setScalar(b.r);
      dummy.updateMatrix();
      this.mesh.setMatrixAt(i, dummy.matrix);
    });
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}
