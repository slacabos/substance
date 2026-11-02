import { Quaternion, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { SpinController } from '../src/interaction/spin';

describe('SpinController', () => {
  it('ignores an ongoing drag while locked and allows dragging again after unlocking', () => {
    const q = new Quaternion();
    const spin = new SpinController(q, 0.2);
    spin.drag(40, 20, 1 / 60);
    spin.setLocked(true);
    const locked = q.clone();
    spin.drag(100, -50, 1 / 60);
    expect(q.equals(locked)).toBe(true);
    expect(spin.angularVelocity.length()).toBe(0);
    spin.setLocked(false);
    spin.drag(100, -50, 1 / 60);
    expect(q.equals(locked)).toBe(false);
  });

  it('turns a locked object back to rest and keeps it there', () => {
    const q = new Quaternion().setFromAxisAngle(new Vector3(1, 1, 0).normalize(), 2);
    const spin = new SpinController(q, 0.2);
    spin.setLocked(true);
    for (let i = 0; i < 300; i++) spin.update(1 / 60, false, 0);
    expect(q.angleTo(new Quaternion())).toBeLessThan(1e-3);
    expect(spin.angularVelocity.length()).toBe(0);
  });

  it('spins idly again once unlocked', () => {
    const q = new Quaternion();
    const spin = new SpinController(q, 0.2);
    spin.setLocked(true);
    spin.update(1 / 60, false, 0);
    spin.setLocked(false);
    for (let i = 0; i < 300; i++) spin.update(1 / 60, false, 0);
    expect(spin.angularVelocity.y).toBeCloseTo(0.2, 2);
  });
});
