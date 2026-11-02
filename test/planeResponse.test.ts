import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { PLANE, planeDirection, planePoint, planePush, planeScale } from '../src/core/chart';
import { PLANE_RESPONSE, planeStreakReach, planeWeatherRecovery } from '../src/core/planeResponse';
import { OceanWeather } from '../src/core/oceanWeather';

describe('plane spray and recovery', () => {
  it('bounds streak aspect even when projection speed is enormous', () => {
    for (const speed of [0, 1, 100, 1e12]) {
      expect(planeStreakReach(0.01, speed, 0.04)).toBeLessThanOrEqual(0.01 * PLANE_RESPONSE.maxStreakAspect);
    }
    expect(planeStreakReach(0.01, 0, 0.04)).toBe(0.01);
  });

  it('preserves orb weather and held gestures, then clears released plane churn faster', () => {
    expect(planeWeatherRecovery(0, false)).toBe(5);
    expect(planeWeatherRecovery(1, true)).toBe(5);
    const weather = new OceanWeather();
    weather.down(new Vector3(0, 0, 1));
    weather.move(new Vector3(0.2, 0, 1).normalize(), 0.1);
    weather.up();
    const peak = weather.peak;
    weather.update(6, planeWeatherRecovery(1, false));
    expect(weather.peak).toBeLessThan(peak * 0.051);
  });

  it('converts advected foam offsets to detail units without losing world velocity', () => {
    for (const [x, z] of [
      [0, 1],
      [0, -20],
      [20, -40],
    ]) {
      const p = planeDirection(x!, z!);
      const velocity = new Vector3(0, 1, 0).cross(p).normalize();
      const dt = 1e-6;
      const moved = planePoint(p.clone().addScaledVector(velocity, dt).normalize())
        .sub(planePoint(p))
        .divideScalar(dt);
      const driftInDetailUnits = planePush(p, velocity).divideScalar(PLANE.detail);
      expect(
        driftInDetailUnits.clone().multiplyScalar(PLANE.detail).distanceTo(moved) / moved.length(),
      ).toBeLessThan(1e-4);
      expect(driftInDetailUnits.length()).toBeCloseTo(planeScale(p) / PLANE.detail, 8);
    }
  });
});
