/** The original, long and springy strand coat, kept independently of the denser dog-like Fur material. */
export const FLUFF = {
  length: 0.26,
  density: 96,
  segments: (coarse: boolean) => (coarse ? 6 : 12),
  points: (coarse: boolean) => (coarse ? 5 : 12),
  colors: {
    root: '#251039',
    tip: '#c59aff',
    accent: '#b991f2',
    highlight: [0.94, 0.87, 1],
  },
  light: [-0.55, 0.75, 0.5],
  /**
   * On the plane's ground: strands drawn shorter than the orb's, a rug's pile a few strands' spacing deep, and a
   * little thinner (grown with the ground's scale they stood like a macro shot of a field); and each leaning its
   * own way, by up to `splay` (as a lean, where 1 is 45°), so the pile splays like a rug's. Each strand is drawn
   * `copies` times over there, every copy on its own root and turned its own way, as each covers far more ground
   * than on the orb (nodes/strandCopies.ts).
   */
  plane: { copies: (coarse: boolean) => (coarse ? 2 : 4), length: 0.28, width: 0.6, splay: 0.9 },
  physics: {
    rate: 240,
    maxSteps: 10,
    gravity: 15,
    springRoot: 1000,
    springTip: 80,
    dampingRatio: 0.3,
    inertia: 0.5,
    maxStep: 0.9,
    brushDrag: 25,
    groomRelax: 0.12,
    combLean: 2.2,
    skin: 0.008,
  },
} as const;

/** Keep the original motion independent of frame timing, with bounded catch-up after a stalled frame. */
export function fluffFrame(dt: number, previousStep: number) {
  if (!(dt > 0) || !Number.isFinite(dt)) return null;
  const { rate, maxSteps } = FLUFF.physics;
  const elapsed = Math.min(dt, maxSteps / rate);
  const steps = Math.min(maxSteps, Math.max(1, Math.ceil(elapsed * rate)));
  const step = elapsed / steps;
  return { dt: elapsed, steps, step, historyScale: step / previousStep };
}
