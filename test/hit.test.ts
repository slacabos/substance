import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { PLANE, planePoint, planeView } from '../src/core/chart';
import { intersectPlane, intersectShape, rayBox, raySphere } from '../src/core/hit';

const fromFront = (x = 0, y = 0) => ({ origin: new Vector3(x, y, 10), dir: new Vector3(0, 0, -1) });

describe('raySphere', () => {
  it('hits the near side of a unit sphere', () => {
    const { origin, dir } = fromFront();
    expect(raySphere(origin, dir, 1)).toBeCloseTo(9);
  });

  it('misses outside the radius', () => {
    const { origin, dir } = fromFront(1.2);
    expect(raySphere(origin, dir, 1)).toBeNull();
  });
});

describe('rayBox', () => {
  it('hits the front face', () => {
    const { origin, dir } = fromFront(0.5, 0.5);
    expect(rayBox(origin, dir, 0.8)).toBeCloseTo(9.2);
  });

  it('misses beside the box', () => {
    const { origin, dir } = fromFront(0.9);
    expect(rayBox(origin, dir, 0.8)).toBeNull();
  });
});

describe('intersectShape', () => {
  it('returns the unit direction to the hit point', () => {
    const { origin, dir } = fromFront();
    const hit = intersectShape(origin, dir, 0, 0, false)!;
    expect(hit.toArray().map((v) => +v.toFixed(6))).toEqual([0, 0, 1]);
  });

  it('uses the box once the morph is mostly cube', () => {
    // x = 0.85 is inside the unit sphere's silhouette but beside the 0.8 cube
    const { origin, dir } = fromFront(0.7, 0.7);
    expect(intersectShape(origin, dir, 0, 0, false)).not.toBeNull();
    expect(intersectShape(origin, dir, 1, 0, false)).not.toBeNull();
    const corner = fromFront(0.85, 0);
    expect(intersectShape(corner.origin, corner.dir, 0, 0, false)).not.toBeNull();
    expect(intersectShape(corner.origin, corner.dir, 1, 0, false)).toBeNull();
  });

  it('counts the fur padding as part of the object', () => {
    const { origin, dir } = fromFront(1.1);
    expect(intersectShape(origin, dir, 0, 0, false)).toBeNull();
    expect(intersectShape(origin, dir, 0, 0.16, false)).not.toBeNull();
  });

  it('snaps a miss to the silhouette when dragging off the edge', () => {
    const { origin, dir } = fromFront(3, 0);
    const hit = intersectShape(origin, dir, 0, 0, true)!;
    expect(hit.x).toBeCloseTo(1);
    expect(hit.length()).toBeCloseTo(1);
  });
});

describe('intersectPlane', () => {
  const view = planeView(1.5);
  const toward = (x: number, z: number) => new Vector3(x, PLANE.groundY, z).sub(view.eye).normalize();

  it('finds the direction whose ground point the ray hits', () => {
    for (const [x, z] of [
      [0, 0],
      [2, -5],
      [-4, 3],
    ] as const) {
      const hit = intersectPlane(view.eye, toward(x, z));
      const ground = planePoint(hit);
      expect(ground.x).toBeCloseTo(x, 6);
      expect(ground.z).toBeCloseTo(z, 6);
    }
  });

  it('lands rays at or above the horizon on the far ring, along their bearing', () => {
    for (const dir of [new Vector3(0, 0, -1), new Vector3(0.3, 0.4, -1).normalize()]) {
      const ground = planePoint(intersectPlane(view.eye, dir));
      expect(Math.hypot(ground.x, PLANE.footZ - ground.z)).toBeCloseTo(PLANE.rhoFar, 6);
      expect(Math.atan2(ground.x, PLANE.footZ - ground.z)).toBeCloseTo(Math.atan2(dir.x, -dir.z), 6);
    }
  });
});
