import { MathUtils, PerspectiveCamera, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import {
  PLANE,
  planeDirection,
  planeNormal,
  planePoint,
  planePush,
  planeScale,
  planeShell,
  planeView,
} from '../src/core/chart';

/** Evenly spread unit directions, as the orb's meshes are. */
const fibonacci = (count: number): Vector3[] => {
  const golden = Math.PI * (3 - Math.sqrt(5));
  return Array.from({ length: count }, (_, i) => {
    const y = 1 - (2 * (i + 0.5)) / count;
    const r = Math.sqrt(1 - y * y);
    return new Vector3(Math.cos(i * golden) * r, y, Math.sin(i * golden) * r);
  });
};

/** Directions on the camera's side of the orb, well inside the chart's clamps. */
const front = fibonacci(4000).filter((p) => p.z > 0.2);

/** Two unit tangents at p with t1 × t2 = p. */
const tangents = (p: Vector3): [Vector3, Vector3] => {
  const t1 = new Vector3(0, 1, 0).cross(p).normalize();
  return [t1, p.clone().cross(t1)];
};

describe('the plane chart', () => {
  it('brings the face toward the camera to the orb’s front-bottom', () => {
    const c = planePoint(new Vector3(0, 0, 1));
    expect(c.x).toBeCloseTo(0);
    expect(c.y).toBeCloseTo(PLANE.groundY);
    expect(c.z).toBeCloseTo(1);
  });

  it('brings the top to the foreground and sends the bottom toward the horizon', () => {
    const top = planePoint(new Vector3(0, 1, 1).normalize());
    const bottom = planePoint(new Vector3(0, -1, 1).normalize());
    expect(top.z).toBeGreaterThan(1);
    expect(bottom.z).toBeLessThan(1);
  });

  it('puts every direction somewhere finite, the back pole included', () => {
    for (const p of [...fibonacci(2000), new Vector3(0, 0, -1)]) {
      const q = planePoint(p);
      expect(Number.isFinite(q.x) && Number.isFinite(q.z)).toBe(true);
      expect(Number.isFinite(planeScale(p))).toBe(true);
    }
  });

  it('inverts', () => {
    for (const p of front) {
      const q = planePoint(p);
      expect(planeDirection(q.x, q.z).distanceTo(p)).toBeLessThan(1e-9);
    }
  });

  it('pushes tangents as the chart moves points', () => {
    const step = 1e-6;
    for (const p of front.slice(0, 200)) {
      for (const t of tangents(p)) {
        const ahead = planePoint(p.clone().addScaledVector(t, step).normalize());
        const expected = ahead.sub(planePoint(p)).divideScalar(step);
        expect(planePush(p, t).distanceTo(expected)).toBeLessThan(1e-4 * expected.length());
      }
    }
  });

  it('is conformal and keeps the ground’s outward side up', () => {
    for (const p of front) {
      const [t1, t2] = tangents(p);
      const a = planePush(p, t1);
      const b = planePush(p, t2);
      const s = planeScale(p);
      expect(a.length() / s).toBeCloseTo(1, 9);
      expect(b.length() / s).toBeCloseTo(1, 9);
      expect(a.dot(b) / (s * s)).toBeCloseTo(0, 9);
      expect(a.cross(b).y).toBeGreaterThan(0);
    }
  });
});

describe('the shell map', () => {
  it('lays the bare orb on the ground with its normals straight up', () => {
    for (const p of front.slice(0, 200)) {
      expect(planeShell(p).y).toBeCloseTo(PLANE.groundY, 12);
      expect(planeNormal(p, p).y).toBeCloseTo(1, 12);
    }
  });

  it('turns a bumpy surface’s normals with the surface', () => {
    // a bump about as steep as a wave, low enough that the height scale's own slope barely shows
    const surface = (p: Vector3) => p.clone().multiplyScalar(1 + 0.003 * Math.sin(40 * p.x + 25 * p.y));
    const step = 1e-5;
    for (const p of front.slice(0, 300)) {
      const [t1, t2] = tangents(p);
      const at = (t: Vector3) => surface(p.clone().addScaledVector(t, step).normalize());
      const a = at(t1).sub(surface(p));
      const b = at(t2).sub(surface(p));
      const n = a.clone().cross(b).normalize();
      const x = surface(p);
      const ma = planeShell(at(t1)).sub(planeShell(x));
      const mb = planeShell(at(t2)).sub(planeShell(x));
      const mapped = planeNormal(p, n);
      expect(Math.abs(mapped.dot(ma.normalize()))).toBeLessThan(0.02);
      expect(Math.abs(mapped.dot(mb.normalize()))).toBeLessThan(0.02);
      expect(mapped.y).toBeGreaterThan(0);
    }
  });
});

describe('the plane view', () => {
  it('looks at its focus, ahead and down', () => {
    for (const aspect of [0.46, 1, 1.78]) {
      const view = planeView(aspect);
      const focus = planePoint(view.focus);
      expect(focus.distanceTo(view.target)).toBeLessThan(1e-6);
      expect(view.target.z).toBeLessThan(view.eye.z);
      expect(view.eye.y - PLANE.groundY).toBeCloseTo(PLANE.eyeHeight);
    }
    expect(planeView(0.46).fov).toBeGreaterThan(planeView(1.78).fov);
  });
});

// The orb's mesh spacing on screen, against how the ground's shows it: the go/no-go for the chart's constants.
describe('the plane’s mesh coverage', () => {
  const directions = fibonacci(53000);
  const spacing = Math.sqrt((8 * Math.PI) / (Math.sqrt(3) * directions.length));
  const coverage = (aspect: number) => {
    const view = planeView(aspect);
    const camera = new PerspectiveCamera(view.fov, aspect, 0.1, 1000);
    camera.position.copy(view.eye);
    camera.lookAt(view.target);
    camera.updateMatrixWorld();
    const forward = camera.getWorldDirection(new Vector3());
    const tanHalf = Math.tan(MathUtils.degToRad(view.fov / 2));
    const rows: number[][] = Array.from({ length: 10 }, () => []);
    let visible = 0;
    for (const p of directions) {
      const q = planePoint(p);
      const ndc = q.clone().project(camera);
      if (Math.abs(ndc.x) > 1 || Math.abs(ndc.y) > 1 || ndc.z > 1) continue;
      if (Math.hypot(q.x, PLANE.footZ - q.z) >= PLANE.rhoFar * 0.999) continue;
      visible++;
      const depth = q.clone().sub(view.eye).dot(forward);
      // the lateral size of one mesh step, in pixels on a screen 1000 pixels tall
      const pixels = (planeScale(p) * spacing * 500) / (tanHalf * depth);
      rows[Math.min(9, Math.floor(((1 - ndc.y) / 2) * 10))]!.push(pixels);
    }
    const median = (row: number[]) => row.sort((a, b) => a - b)[row.length >> 1] ?? Infinity;
    return { visible: visible / directions.length, rows: rows.map(median) };
  };

  it('spreads a portrait phone’s mesh within a few times the orb’s spacing', () => {
    const { visible, rows } = coverage(0.46);
    expect(visible).toBeGreaterThan(0.25);
    // the lower 60% of the screen; nearer the horizon the haze takes over
    for (const pixels of rows.slice(4)) expect(pixels).toBeLessThan(16);
  });

  it('spreads a landscape screen’s mesh within a few times the orb’s spacing', () => {
    const { visible, rows } = coverage(1.78);
    expect(visible).toBeGreaterThan(0.55);
    for (const pixels of rows.slice(4)) expect(pixels).toBeLessThan(16);
  });
});
