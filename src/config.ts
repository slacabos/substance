/** Ripple / dent ring-buffer size; it's also the shader's uniform array length. */
export const MAX_IMPULSES = 16;
export const FUR = {
  length: 0.26,
  density: 96,
  /** Ribbon segments per strand; more makes bent hair curve more smoothly. */
  segments: (coarse: boolean) => (coarse ? 6 : 12),
  /** Simulated points per strand, root included; the ribbon curves smoothly through them. */
  points: (coarse: boolean) => (coarse ? 5 : 12),
} as const;

export const OCEAN_FFT = {
  /** Texels per side of each wave tile. */
  size: (coarse: boolean) => (coarse ? 128 : 256),
  /**
   * Wave cascades, longest first, as a lattice step [a, b] per cube face: a quarter turn of the orb holds
   * sqrt(a² + b²) tiles, turned by atan(b / a). Whole numbers keep the waves seamless around the equator.
   */
  cascades: (coarse: boolean): readonly (readonly [number, number])[] =>
    coarse
      ? [
          [1, 0],
          [8, 3],
        ]
      : [
          [1, 0],
          [4, 1],
          [17, -6],
        ],
} as const;

/**
 * Desktop renders above screen resolution while the canvas stays under this many pixels, which smooths the fur's
 * fine strands. Phones already render at 2-3x and are left alone.
 */
export const SUPERSAMPLE_PIXELS = 3840 * 2160; // 4K

/** Radius the camera keeps in view: cube corner plus fur. */
export const FIT_RADIUS = 1.45;
