import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import type { SprayGround } from '../src/materials/oceanSim';
import { OceanSpray } from '../src/materials/oceanSpray';

describe('ocean spray', () => {
  it('returns to the surface and leaves foam', () => {
    const deposits: number[] = [];
    const ground: SprayGround = {
      groundHeight: () => 0,
      depositFoam: (_dir, amount) => deposits.push(amount),
    };
    const spray = new OceanSpray(true);
    spray.emitCrest(new Vector3(0, 0, 1), new Vector3(0, 1, 0), 1, 30);
    for (let i = 0; i < 90; i++) spray.update(1 / 60, ground);
    expect(deposits.length).toBeGreaterThan(0);
    expect(deposits.every((amount) => amount > 0)).toBe(true);
    spray.clear();
  });
});
