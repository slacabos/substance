export interface DeviceProfile {
  /** Touch-first device: lighter geometry and simpler fur strands. */
  coarse: boolean;
  reduceMotion: boolean;
}

export function detectDevice(): DeviceProfile {
  return {
    coarse: matchMedia('(pointer: coarse)').matches,
    reduceMotion: matchMedia('(prefers-reduced-motion: reduce)').matches,
  };
}
