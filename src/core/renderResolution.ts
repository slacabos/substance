import { SUPERSAMPLE_PIXELS } from '../config';

/** Drawing-buffer policy; CSS size and input coordinates never change. */
export function renderPixelRatio(
  width: number,
  height: number,
  dpr: number,
  supersample: boolean,
  coarse: boolean,
  plane: boolean,
): number {
  const area = Math.max(1, width * height);
  if (coarse && plane) return Math.min(dpr, 1.5, Math.sqrt(1_500_000 / area));
  const budget = Math.sqrt(SUPERSAMPLE_PIXELS / area);
  return Math.min(2, supersample ? Math.max(dpr, budget) : dpr);
}
