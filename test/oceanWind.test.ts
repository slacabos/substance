import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { dispersion, OCEAN_WIND_ANGLE } from '../src/core/oceanFFT';
import { GustField, oceanWind, relaxGain, WIND, windAt } from '../src/core/oceanWind';

describe('ocean wind', () => {
  it('wanders around the tuned wind without running away', () => {
    const angles: number[] = [];
    let speed = 0;
    const samples = 40000;
    for (let i = 0; i < samples; i++) {
      const wind = oceanWind(i * 0.5, 0.9);
      angles.push(wind.angle);
      speed += wind.speed;
      expect(Math.abs(wind.angle - OCEAN_WIND_ANGLE)).toBeLessThanOrEqual(WIND.veer);
      expect(wind.speed).toBeGreaterThanOrEqual(0);
      expect(wind.gustiness).toBeGreaterThanOrEqual(0);
      expect(wind.gustiness).toBeLessThanOrEqual(1);
    }
    const mean = angles.reduce((a, b) => a + b) / samples;
    const spread = Math.sqrt(angles.reduce((a, b) => a + (b - mean) ** 2, 0) / samples);
    expect(Math.abs(mean - OCEAN_WIND_ANGLE)).toBeLessThan(0.08);
    // it really does swing, by tens of degrees
    expect(spread).toBeGreaterThan(0.12);
    expect(spread).toBeLessThan(0.5);
    // gusts only add, so the mean sits a little above the tuned speed
    expect(speed / samples / 0.9).toBeGreaterThan(0.97);
    expect(speed / samples / 0.9).toBeLessThan(1.15);
  });

  it('changes smoothly, the same way every time, and stays calm without wind', () => {
    for (let t = 0; t < 300; t += 0.37) {
      const a = oceanWind(t, 0.9);
      const b = oceanWind(t + 0.05, 0.9);
      expect(Math.abs(b.angle - a.angle)).toBeLessThan(0.02);
      expect(Math.abs(b.speed - a.speed)).toBeLessThan(0.05);
      expect(oceanWind(t, 0.9)).toEqual(a);
      expect(oceanWind(t, 0).speed).toBe(0);
    }
  });

  it('turns short waves toward a new wind long before the swell', () => {
    const short = dispersion((2 * Math.PI) / 0.02);
    const swell = dispersion((2 * Math.PI) / 0.35);
    let g = [1, 1];
    // the wind swings away: both waves' targets drop to nothing, then a new direction raises them
    for (let t = 0; t < 4; t += 1 / 60)
      g = [relaxGain(g[0]!, 0, 1 / 60, short), relaxGain(g[1]!, 0, 1 / 60, swell)];
    expect(g[0]).toBeLessThan(0.55);
    expect(g[1]).toBeGreaterThan(0.85);
    // and growing is quicker than dying away
    let grow = 0;
    let decay = 1;
    for (let t = 0; t < 2; t += 1 / 60) {
      grow = relaxGain(grow, 1, 1 / 60, short);
      decay = relaxGain(decay, 0, 1 / 60, short);
    }
    expect(1 - grow).toBeLessThan(decay);
  });

  it('blows along the orb, east to north by its angle, dying away at the poles', () => {
    const p = new Vector3(1, 0, 0);
    // east at +x is -z (up x p), north is up
    expect(windAt(p, 0).toArray()).toEqual([0, 0, -1]);
    expect(windAt(p, Math.PI / 2).distanceTo(new Vector3(0, 1, 0))).toBeLessThan(1e-12);
    for (const q of [new Vector3(0.3, 0.8, -0.5).normalize(), new Vector3(-0.6, -0.2, 0.77).normalize()]) {
      const w = windAt(q, 1.1);
      expect(Math.abs(w.dot(q))).toBeLessThan(1e-12);
      expect(w.length()).toBeCloseTo(Math.sqrt(1 - q.y * q.y), 12);
    }
    expect(windAt(new Vector3(0, 1, 0), 0.4).length()).toBe(0);
  });

  it("sweeps cat's paws downwind and fades them in and out", () => {
    const field = new GustField(3);
    const wind = { angle: 0.3, speed: 1, gustiness: 1 };
    const gust = field.gusts[0]!;
    // keep this one away from the poles and its respawn
    gust.center.set(1, 0.1, 0.2).normalize();
    gust.age = 0;
    gust.life = 100;
    const start = gust.center.clone();
    const downwind = windAt(start, wind.angle).normalize();
    field.update(1, wind);
    const moved = gust.center.clone().sub(start);
    expect(moved.length()).toBeCloseTo(WIND.gustTravel, 2);
    expect(moved.normalize().dot(downwind)).toBeGreaterThan(0.99);
    for (let i = 0; i < 2000; i++) {
      field.update(0.05, wind);
      for (const g of field.gusts) {
        expect(g.center.length()).toBeCloseTo(1, 9);
        expect(field.intensity(g)).toBeGreaterThanOrEqual(0);
        expect(field.intensity(g)).toBeLessThanOrEqual(1);
      }
    }
  });
});
