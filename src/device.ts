export interface DeviceProfile {
  /** Touch-first device: lighter geometry and fewer fur shells. */
  coarse: boolean;
  reduceMotion: boolean;
}

export function detectDevice(): DeviceProfile {
  return {
    coarse: matchMedia('(pointer: coarse)').matches,
    reduceMotion: matchMedia('(prefers-reduced-motion: reduce)').matches,
  };
}
