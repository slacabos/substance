import { Quaternion, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { PLANE, planeDirection, planePoint } from '../src/core/chart';
import {
  UNFURL,
  unfurlCameraWeight,
  unfurlPoint,
  unfurlRadius,
  unfurlRotationWeight,
  unfurlSolidWeight,
  unfurlVisible,
} from '../src/core/transition';
import { SpinController } from '../src/interaction/spin';
import { ShapeMorph } from '../src/scene/shape';

describe('continuous unfurl', () => {
  it('keeps the cube recognizable until opening, and returns directly to a cube', () => {
    const morph = new ShapeMorph();
    morph.set('cube', true);
    morph.set('plane', false);
    morph.update(0.05);
    expect(morph.value).toBeCloseTo(1, 8);
    for (let i = 0; i < 240; i++) morph.update(1 / 60);
    expect(morph.plane).toBe(1);
    expect(morph.value).toBe(0);
    morph.set('cube', false);
    morph.update(0.05);
    expect(morph.value).toBe(0);
    for (let i = 0; i < 240; i++) morph.update(1 / 60);
    expect(morph.value).toBeCloseTo(1, 8);
    expect(morph.plane).toBe(0);
  });

  it('keeps the early rim bounded and reaches the unchanged full territory', () => {
    for (const [x, z] of [
      [0, 1],
      [20, -40],
      [-10, -100],
      [0, -140],
    ]) {
      const point = planePoint(planeDirection(x!, z!));
      expect(unfurlPoint(point, 1).distanceTo(point)).toBeLessThan(1e-9);
      for (const p of [0.1, 0.25, 0.5, 0.75]) {
        const rim = unfurlPoint(point, p);
        expect(Math.hypot(rim.x, rim.z - 1)).toBeLessThanOrEqual(unfurlRadius(p) + 1e-9);
        expect(rim.y).toBe(PLANE.groundY);
      }
    }
    expect(unfurlRadius(0.25)).toBeLessThan(3);
  });

  it('uses continuous reversible camera, solid and orientation curves with exact endpoints', () => {
    expect(unfurlCameraWeight(0)).toBe(0);
    expect(unfurlCameraWeight(1)).toBe(1);
    expect(unfurlCameraWeight(0.1)).toBe(0);
    expect(unfurlRotationWeight(UNFURL.alignUntil)).toBe(1);
    expect(unfurlSolidWeight(1, 0)).toBe(1);
    expect(unfurlSolidWeight(1, 1)).toBe(0);
    let camera = 0,
      radius: number = UNFURL.startRadius;
    for (let i = 0; i <= 100; i++) {
      const p = i / 100;
      expect(unfurlCameraWeight(p)).toBeGreaterThanOrEqual(camera);
      expect(unfurlRadius(p)).toBeGreaterThanOrEqual(radius);
      camera = unfurlCameraWeight(p);
      radius = unfurlRadius(p);
    }
  });

  it('opens only the hidden pole and leaves every valid ground direction intact', () => {
    expect(unfurlVisible(new Vector3(0, 0, -1), 0)).toBe(true);
    expect(unfurlVisible(new Vector3(0, 0, -1), 1)).toBe(false);
    expect(unfurlRotationWeight(UNFURL.seamFrom)).toBeGreaterThanOrEqual(0.5);
    for (const rho of [PLANE.rhoMin, 1, 6, 60, PLANE.rhoFar]) {
      for (const phi of [-PLANE.phiMax, -0.5, 0, 0.5, PLANE.phiMax]) {
        const p = planeDirection(rho * Math.sin(phi), PLANE.footZ - rho * Math.cos(phi));
        expect(unfurlVisible(p, 1)).toBe(true);
      }
    }
    const back = new Vector3(0, 0, -1),
      point = planePoint(back);
    expect(unfurlPoint(point, 0.5, back).y).toBeLessThan(point.y - 2);
    expect(unfurlPoint(point, 1, back).y).toBe(point.y);
  });

  it('coordinates orientation through reversals and unlocks only after reforming', () => {
    const original = new Quaternion().setFromAxisAngle(new Vector3(1, 1, 0).normalize(), 1.7);
    const q = original.clone();
    const spin = new SpinController(q, 0.2);
    spin.setPlane(0, true);
    expect(q.angleTo(original)).toBeLessThan(1e-7);
    spin.setPlane(0.14, true);
    const partial = q.clone();
    spin.setPlane(1, true);
    expect(q.angleTo(new Quaternion())).toBeLessThan(1e-7);
    spin.setPlane(1, false);
    for (let i = 0; i < 60; i++) spin.update(1 / 60, false, Infinity);
    expect(q.angleTo(new Quaternion())).toBeLessThan(1e-7);
    spin.setPlane(0.14, false);
    expect(q.angleTo(partial)).toBeLessThan(1e-7);
    spin.setPlane(0.14, true);
    expect(q.angleTo(partial)).toBeLessThan(1e-7);
    spin.drag(100, 50, 1 / 60);
    expect(q.angleTo(partial)).toBeLessThan(1e-7);
    spin.setPlane(0, false);
    expect(q.angleTo(original)).toBeLessThan(1e-7);
    spin.drag(100, 50, 1 / 60);
    expect(q.angleTo(original)).toBeGreaterThan(0.1);
  });

  it('jumps to the coordinated endpoints for reduced motion', () => {
    const morph = new ShapeMorph();
    const q = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), 1);
    const spin = new SpinController(q, 0.2),
      original = q.clone();
    morph.set('cube', true);
    morph.set('plane', true);
    morph.update(0);
    spin.setPlane(morph.plane, true);
    expect(morph.value).toBe(0);
    expect(q.angleTo(new Quaternion())).toBeLessThan(1e-7);
    morph.set('cube', true);
    morph.update(0);
    spin.setPlane(morph.plane, false);
    expect(morph.value).toBe(1);
    expect(q.angleTo(original)).toBeLessThan(1e-7);
  });
});
