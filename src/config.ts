/** Ring-buffer sizes. These become GLSL array lengths through material defines. */
export const MAX_IMPULSES = 16;
export const MAX_BRUSHES = 12;

export const FUR = {
  length: 0.26,
  density: 64,
  /** Ribbon segments per strand; more makes bent hair curve more smoothly. */
  segments: (coarse: boolean) => (coarse ? 6 : 10),
} as const;

/**
 * Desktop renders above screen resolution while the canvas stays under this many pixels, which smooths the fur's
 * fine strands. Phones already render at 2-3x and are left alone.
 */
export const SUPERSAMPLE_PIXELS = 4.2e6;

/** Radius the camera keeps in view: cube corner plus fur. */
export const FIT_RADIUS = 1.45;
