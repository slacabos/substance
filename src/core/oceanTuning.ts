export interface OceanTuning {
  windSpeed: number;
  peakWavelength: number;
  amplitude: number;
  choppiness: number;
  foam: number;
}

export type OceanTuningValue = keyof OceanTuning;

export const OCEAN_DEFAULTS: Readonly<OceanTuning> = {
  windSpeed: 0.9,
  peakWavelength: 0.23,
  amplitude: 0.95,
  choppiness: 2,
  foam: 1.4,
};

export const OCEAN_LIMITS: Record<OceanTuningValue, readonly [number, number]> = {
  windSpeed: [0, 3],
  peakWavelength: [0.14, 0.65],
  amplitude: [0, 2],
  choppiness: [0, 2],
  foam: [0, 2.5],
};

const KEYS: Record<OceanTuningValue, string> = {
  windSpeed: 'oceanWind',
  peakWavelength: 'oceanWavelength',
  amplitude: 'oceanAmplitude',
  choppiness: 'oceanChop',
  foam: 'oceanFoam',
};

export const OCEAN_VALUES = Object.keys(KEYS) as OceanTuningValue[];

/** Around 8k vertices on touch devices and 24k on desktops. */
export function oceanDetail(coarse: boolean): number {
  return coarse ? 28 : 48;
}

export function readOceanTuning(search: string): OceanTuning {
  const query = new URLSearchParams(search);
  const tuning: OceanTuning = { ...OCEAN_DEFAULTS };
  for (const name of OCEAN_VALUES) {
    const raw = query.get(KEYS[name]);
    if (raw === null || raw.trim() === '') continue;
    const value = Number(raw);
    if (!Number.isFinite(value)) continue;
    const [min, max] = OCEAN_LIMITS[name];
    tuning[name] = Math.min(max, Math.max(min, value));
  }
  return tuning;
}

/**
 * The query string with the tuning written back on top of `base` (other parameters are kept), leaving out
 * values at their defaults so the plain URL stays clean. Returns '' when nothing needs saying.
 */
export function oceanTuningQuery(tuning: Readonly<OceanTuning>, base = ''): string {
  const query = new URLSearchParams(base);
  for (const name of OCEAN_VALUES) {
    const value = Math.round(tuning[name] * 1000) / 1000;
    if (value === OCEAN_DEFAULTS[name]) query.delete(KEYS[name]);
    else query.set(KEYS[name], String(value));
  }
  const text = query.toString();
  return text ? `?${text}` : '';
}
