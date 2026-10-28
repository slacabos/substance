import { IcosahedronGeometry, Vector3 } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import type { WebGPURenderer } from 'three/webgpu';
import { describe, expect, it } from 'vitest';
import {
  sampleShoreSurf,
  SHELF,
  SHELF_EDGE,
  shelfDepth,
  shelfLag,
  SHORE_SITES,
  shoreDistance,
  shoreElevation,
  shoreRadius,
  shoreSurfPhase,
  shoreSwell,
  SURF_SET_WAVES,
  SURF_TRAINS,
  shoreSurfAt,
  surfBend,
  surfSet,
} from '../src/core/oceanShores';
import { OCEAN_DEFAULTS } from '../src/core/oceanTuning';
import { OceanShores } from '../src/materials/oceanShores';
import { OceanSim } from '../src/materials/oceanSim';

const sphere = (detail: number) => {
  const source = new IcosahedronGeometry(1, detail);
  source.deleteAttribute('uv');
  return mergeVertices(source);
};

describe('coastal ocean', () => {
  it('covers about half the orb with one beach above the largest swells, its sand shelving into the sea', () => {
    // one stretch of land over about half the orb: a beach, not a continent and islands
    expect(SHORE_SITES).toHaveLength(1);
    let land = 0;
    const samples = 4000;
    for (let i = 0; i < samples; i++) {
      const z = 1 - (2 * (i + 0.5)) / samples;
      const a = i * 2.399963;
      const r = Math.sqrt(1 - z * z);
      if (shoreDistance(new Vector3(r * Math.cos(a), z, r * Math.sin(a))) < 0) land++;
    }
    expect(land / samples).toBeGreaterThan(0.4);
    expect(land / samples).toBeLessThan(0.6);
    const peaks = SHORE_SITES.map((site) => shoreElevation(site, site.center));
    for (const site of SHORE_SITES) expect(shoreDistance(site.center)).toBeLessThan(0);
    for (const peak of peaks) expect(peak).toBeGreaterThan(0.025);
    expect(Math.max(...peaks)).toBeLessThan(0.06);
    // a beach climbs gently from the waterline: no step out of the sea
    for (const site of SHORE_SITES) {
      const ashore = (d: number) => {
        // due east of the centre (bearing 0), the waterline is shoreRadius(site, 0) out
        const angle = shoreRadius(site, 0) + d;
        const dir = site.center.clone().multiplyScalar(Math.cos(angle));
        return shoreElevation(site, dir.addScaledVector(site.east, Math.sin(angle)).normalize());
      };
      expect(ashore(0.002)).toBeLessThan(0.006);
      expect(ashore(-site.beach * 0.5)).toBeLessThan(0.02);
    }
    expect(shoreDistance(SHORE_SITES[0]!.center.clone().negate())).toBeGreaterThan(0);
    // the sim's buffers the land reads the sea from; no GPU in tests
    const sim = new OceanSim({ compute: () => {} } as unknown as WebGPURenderer, sphere(6));
    const shores = new OceanShores(true, sim.renderBuffers());
    const position = shores.mesh.geometry.getAttribute('position');
    const before = position.getY(0);
    shores.updateMorph(1);
    expect(position.getY(0)).not.toBe(before);
  });

  it('shelves the seabed gently out to the open sea, over a sandbar with a trough inside it', () => {
    const { bar } = SHELF;
    // shallow a long way out: the shallows are what make it a beach
    expect(shelfDepth(0.1)).toBeLessThan(0.012);
    expect(SHELF_EDGE).toBeGreaterThan(0.4);
    // deepening seaward everywhere but over the bar's inner flank
    for (let d = 0.02; d < SHELF_EDGE; d += 0.005) {
      const onBar = d > bar.at - 2 * bar.width && d < bar.at;
      if (!onBar) expect(shelfDepth(d + 0.005)).toBeGreaterThanOrEqual(shelfDepth(d));
    }
    // the bar stands shallower than the trough inside it, and the water is deep enough over it to break on
    const trough = Math.max(...[0.06, 0.08, 0.1, 0.11].map(shelfDepth));
    expect(shelfDepth(bar.at)).toBeLessThan(trough);
    expect(shelfDepth(bar.at)).toBeGreaterThan(0.004);
  });

  const swell = shoreSwell(OCEAN_DEFAULTS.peakWavelength);
  const surfAt = (distance: number, time: number, facing = 1) => {
    const out = { height: 0, breaking: 0 };
    const phase = shoreSurfPhase(0, distance, shelfLag(distance), swell, swell.omega * time);
    sampleShoreSurf(phase, distance, facing, 1, out);
    return out;
  };

  it('runs swell in toward the bank, crowding it on the shelf, and breaks it there', () => {
    // a crest is a point of fixed phase; the phase falls away from the shore, so find it by bisection
    const phaseAt = (d: number, time: number) => shoreSurfPhase(0, d, shelfLag(d), swell, swell.omega * time);
    const crestAt = (phase: number, time: number) => {
      let [near, far] = [0, 0.1];
      for (let i = 0; i < 60; i++) {
        const mid = (near + far) / 2;
        if (phaseAt(mid, time) > phase) near = mid;
        else far = mid;
      }
      return near;
    };
    const target = phaseAt(0.05, 0);
    expect(crestAt(target, 1)).toBeLessThan(0.05 - 0.01);
    // and it slows as it comes in
    const outer = 0.05 - crestAt(target, 1);
    const inner = crestAt(phaseAt(0.01, 0), 0) - crestAt(phaseAt(0.01, 0), 1);
    expect(inner).toBeLessThan(outer);
    // a crest gains phase faster per unit distance near the bank than out at sea: it has slowed and shortened
    const slope = (d: number) =>
      Math.abs(
        shoreSurfPhase(0, d + 1e-4, shelfLag(d + 1e-4), swell, 0) -
          shoreSurfPhase(0, d, shelfLag(d), swell, 0),
      ) / 1e-4;
    const open = SHELF_EDGE + 0.05;
    expect(slope(0.005)).toBeGreaterThan(2 * slope(open));
    expect(slope(open)).toBeCloseTo(swell.k, 3);
    const most = (d: number, key: 'height' | 'breaking') => {
      let peak = 0;
      for (let time = 0; time < 10; time += 0.05) peak = Math.max(peak, surfAt(d, time)[key]);
      return peak;
    };
    expect(most(0.012, 'breaking')).toBeGreaterThan(0.9);
    // it breaks on the bar too, before it reaches the beach; between them it runs on as whitewater
    expect(most(SHELF.bar.at, 'breaking')).toBeGreaterThan(0.6);
    expect(most(SHELF.bar.at + 0.08, 'breaking')).toBeLessThan(0.2);
    // and it never stands taller than the water lets it, so no trough bares the seabed
    for (const d of [0.01, 0.05, SHELF.bar.at, 0.25]) {
      let trough = 0;
      for (let time = 0; time < 10; time += 0.05) trough = Math.min(trough, surfAt(d, time).height);
      expect(-trough).toBeLessThan(shelfDepth(d) * 0.5);
    }
    expect(surfAt(-0.01, 3)).toEqual({ height: 0, breaking: 0 });
    expect(surfAt(0.5, 3)).toEqual({ height: 0, breaking: 0 });
  });

  it('brings its crests straight in at the beach, nearly parallel to it, not sweeping along it', () => {
    for (const train of SURF_TRAINS) {
      // along the beach a crest is late by only its slight slant and bend: a quarter turn round the land is a
      // few waves at most
      const phase = (bearing: number, d: number) => shoreSurfPhase(bearing, d, shelfLag(d), swell, 0, train);
      const along = phase(Math.PI / 2, 0.05) - phase(0, 0.05);
      const bend = surfBend(train, Math.PI / 2) - surfBend(train, 0);
      expect(along).toBeCloseTo((train.slant * Math.PI) / 2 + bend, 6);
      // while across it, from the bar to the beach, it's several waves
      expect(phase(0, 0) - phase(0, SHELF.bar.at)).toBeGreaterThan(2 * Math.PI);
      // a whole number of crests round the land, so the phase joins up all the way round
      expect(Number.isInteger(train.slant)).toBe(true);
      expect(phase(2 * Math.PI, 0.05) - phase(0, 0.05)).toBeCloseTo(2 * Math.PI * train.slant, 6);
    }
  });

  it('comes in irregular waves: not one sine, of many sizes, breaking at different times along the beach', () => {
    const period = (2 * Math.PI) / swell.omega;
    const out = { height: 0, breaking: 0 };
    // the surf at a point on the bar over a minute, every trains' clock running at its own frequency
    const at = (bearing: number, time: number) => {
      const clocks = SURF_TRAINS.map((t) => swell.omega * t.ratio * time);
      shoreSurfAt(bearing, SHELF.bar.at, shelfLag(SHELF.bar.at), swell, clocks, 1, out);
      return out.height;
    };
    const step = 0.05;
    const series = (bearing: number) => Array.from({ length: 1200 }, (_, k) => at(bearing, k * step));
    const here = series(0);
    // a single sine would repeat itself exactly a period on; this barely does
    const lag = Math.round(period / step);
    const mean = here.reduce((a, b) => a + b, 0) / here.length;
    let same = 0;
    let all = 0;
    for (let k = 0; k + lag < here.length; k++) {
      same += (here[k]! - mean) * (here[k + lag]! - mean);
      all += (here[k]! - mean) ** 2;
    }
    expect(same / all).toBeLessThan(0.8);
    // waves of many sizes: the crests a period apart differ a lot
    const crests: number[] = [];
    for (let k = 1; k + 1 < here.length; k++)
      if (here[k]! > here[k - 1]! && here[k]! >= here[k + 1]! && here[k]! > 0) crests.push(here[k]!);
    expect(Math.max(...crests) / Math.min(...crests)).toBeGreaterThan(1.8);
    // and a stretch of beach a little way along breaks to its own timing
    const there = series(0.6);
    let both = 0;
    for (let k = 0; k < here.length; k++) both += (here[k]! - mean) * (there[k]! - mean);
    expect(Math.abs(both / all)).toBeLessThan(0.7);
  });

  it('breaks hardest on shores facing the swell, and keeps a steady pace however long it runs', () => {
    let exposed = 0;
    let lee = 0;
    for (let time = 0; time < 10; time += 0.05) {
      exposed = Math.max(exposed, surfAt(0.012, time, 1).breaking);
      lee = Math.max(lee, surfAt(0.012, time, -1).breaking);
    }
    expect(lee).toBeLessThan(0.25 * exposed);
    // the clock only ever adds omega per second, never time times a changing speed
    for (const time of [1, 100, 10000]) {
      const step =
        shoreSurfPhase(0, 0.01, shelfLag(0.01), swell, swell.omega * (time + 1)) -
        shoreSurfPhase(0, 0.01, shelfLag(0.01), swell, swell.omega * time);
      expect(step).toBeCloseTo(-swell.omega, 6);
    }
  });

  it('brings the surf in sets: groups of bigger waves, then a lull, about a minute apart', () => {
    // the biggest breaker at a point on the bank in each wave period, over a few sets
    const period = (2 * Math.PI) / swell.omega;
    const peaks: number[] = [];
    for (let wave = 0; wave < 4 * SURF_SET_WAVES; wave++) {
      let peak = 0;
      for (let t = wave * period; t < (wave + 1) * period; t += 0.05) {
        const spatial = shoreSurfPhase(0, 0.012, shelfLag(0.012), swell, 0);
        const out = { height: 0, breaking: 0 };
        const clock = swell.omega * t;
        sampleShoreSurf(spatial - clock, 0.012, 1, surfSet(spatial, clock / SURF_SET_WAVES, 0), out);
        peak = Math.max(peak, out.breaking);
      }
      peaks.push(peak);
    }
    const sorted = [...peaks].sort((a, b) => a - b);
    expect(sorted.at(-1)! / sorted[0]!).toBeGreaterThan(2);
    // and they come round again: the strongest waves of each set repeat every SURF_SET_WAVES
    for (let wave = 0; wave + SURF_SET_WAVES < peaks.length; wave++)
      expect(peaks[wave + SURF_SET_WAVES]).toBeCloseTo(peaks[wave]!, 1);
  });
});
