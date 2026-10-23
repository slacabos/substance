import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { OceanWeather, weatherRadius } from '../src/core/oceanWeather';

describe('ocean weather', () => {
  it('keeps a stationary press calm and develops rough weather with movement', () => {
    const weather = new OceanWeather();
    weather.down(new Vector3(0, 0, 1));
    expect(weather.peak).toBe(0);
    expect(weather.strengthAt(new Vector3(0, 0, 1))).toBe(0);
    expect(weather.strengthAt(new Vector3(0, 0, -1))).toBe(0);
    weather.move(new Vector3(0.2, 0, 0.98).normalize(), 0.1);
    expect(weather.peak).toBeGreaterThan(0);
    expect(weather.patches[0]!.center.x).toBeGreaterThan(0);
    weather.up();
  });

  it('spreads and calms over time, then frees the patch slot', () => {
    const weather = new OceanWeather();
    weather.down(new Vector3(0, 0, 1));
    weather.move(new Vector3(0.2, 0, 0.98).normalize(), 0.1);
    const initial = weather.peak;
    expect(weatherRadius(0)).toBeCloseTo(0.35);
    expect(weatherRadius(5)).toBeCloseTo(0.65);
    expect(weatherRadius(20)).toBeCloseTo(0.65);
    weather.update(5);
    expect(weather.peak).toBeCloseTo(initial / Math.E);
    weather.update(20);
    expect(weather.peak).toBe(0);
  });

  it('keeps at most four active patches and replaces the weakest', () => {
    const weather = new OceanWeather();
    for (let i = 0; i < 5; i++) {
      weather.down(new Vector3(0.2 + i * 0.12, 0, 1).normalize());
      weather.move(new Vector3(0.26 + i * 0.12, 0, 1).normalize(), 0.03);
      weather.up();
      weather.update(0.5);
    }
    expect(weather.patches.filter((patch) => patch.strength > 0)).toHaveLength(4);
    expect(weather.patches.some((patch) => patch.center.x < 0.25)).toBe(false);
    expect(weather.patches.some((patch) => patch.center.x > 0.55)).toBe(true);
  });

  it('can reach rough weather after sustained fast stirring', () => {
    const weather = new OceanWeather();
    weather.down(new Vector3(0, 0, 1));
    for (let i = 1; i <= 100; i++) {
      const angle = i * 0.04;
      weather.move(new Vector3(Math.sin(angle), 0, Math.cos(angle)), 0.02);
      weather.update(0.02);
    }
    expect(weather.peak).toBeGreaterThan(0.9);
    weather.up();
  });
});
