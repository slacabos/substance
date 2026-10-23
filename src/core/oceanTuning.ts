export type OceanLod = 'low' | 'medium' | 'high';

export interface OceanTuning {
  windSpeed: number;
  peakWavelength: number;
  amplitude: number;
  choppiness: number;
  foam: number;
  lod: OceanLod;
}

export type OceanTuningValue = Exclude<keyof OceanTuning, 'lod'>;

export const OCEAN_DEFAULTS: Readonly<OceanTuning> = {
  windSpeed: 1,
  peakWavelength: 0.33,
  amplitude: 1,
  choppiness: 1,
  foam: 1.3,
  lod: 'medium',
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

export function oceanDetail(lod: OceanLod): number {
  return lod === 'low' ? 28 : lod === 'high' ? 72 : 44;
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
  const lod = query.get('oceanLod');
  if (lod === 'low' || lod === 'medium' || lod === 'high') tuning.lod = lod;
  return tuning;
}

export function oceanTuningQuery(search: string, tuning: OceanTuning): string {
  const query = new URLSearchParams(search);
  for (const name of OCEAN_VALUES) {
    if (tuning[name] === OCEAN_DEFAULTS[name]) query.delete(KEYS[name]);
    else query.set(KEYS[name], String(tuning[name]));
  }
  if (tuning.lod === OCEAN_DEFAULTS.lod) query.delete('oceanLod');
  else query.set('oceanLod', tuning.lod);
  const result = query.toString();
  return result ? `?${result}` : '';
}
