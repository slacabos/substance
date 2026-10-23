import { Color } from 'three';

export type SettingId = 'studio' | 'dusk' | 'night';

/** Picker order, also what the B key cycles through. */
export const SETTING_ORDER: readonly SettingId[] = ['studio', 'dusk', 'night'];

/**
 * How a setting paints the backdrop. Colours are linear and go through ACES tone mapping, which is why
 * the near-whites have extra blue: it keeps them neutral on screen.
 */
export interface SettingLook {
  /** Upper wall. */
  wall: Color;
  /** Floor and lower wall. */
  floor: Color;
  /** Band where wall meets floor. */
  horizon: Color;
  /** Soft light above and behind the object. */
  halo: Color;
  /** How much of the material's coloured light pool shows on the floor. */
  glow: number;
  /** Contact shadow under the object. */
  shadow: number;
}

export const SETTINGS: Record<SettingId, SettingLook> = {
  studio: {
    wall: new Color(0.84, 0.97, 1.24),
    floor: new Color(0.48, 0.55, 0.68),
    horizon: new Color(0, 0, 0),
    halo: new Color(0.22, 0.25, 0.32),
    glow: 0,
    shadow: 0.3,
  },
  dusk: {
    wall: new Color(0.16, 0.12, 0.3),
    floor: new Color(0.05, 0.025, 0.04),
    horizon: new Color(0.9, 0.36, 0.18),
    halo: new Color(0.55, 0.32, 0.4),
    glow: 0.7,
    shadow: 0.18,
  },
  night: {
    wall: new Color(0.04, 0.05, 0.085),
    floor: new Color(0.006, 0.008, 0.016),
    horizon: new Color(0.035, 0.05, 0.09),
    halo: new Color(0.49, 0.56, 0.7),
    glow: 1,
    shadow: 0,
  },
};

/** Browser chrome colour per setting: the header's surface, so the status bar runs into it. */
export const THEME_COLORS: Record<SettingId, string> = {
  studio: '#e4e8ee',
  dusk: '#1b202b',
  night: '#1b202b',
};
