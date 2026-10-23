/** Ring-buffer sizes. These become GLSL array lengths through material defines. */
export const MAX_IMPULSES = 16;
export const MAX_BRUSHES = 12;

export const FUR = {
  length: 0.26,
  density: 64,
  shells: (coarse: boolean) => (coarse ? 40 : 60),
} as const;

/** Radius the camera keeps in view: cube corner plus fur. */
export const FIT_RADIUS = 1.45;
