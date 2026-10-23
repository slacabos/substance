import type { MaterialId, Readout, ShapeId } from '../materials/types';
import { THEME_COLORS, type SettingId } from '../scene/settings';

const MATERIAL_KEYS: Record<string, MaterialId> = { '1': 'water', '2': 'ocean', '3': 'slime', '4': 'fur' };
const SHAPE_KEYS: Record<string, ShapeId> = { o: 'orb', c: 'cube' };

export interface ControlHandlers {
  onMaterial(id: MaterialId): void;
  onShape(id: ShapeId): void;
  onSetting(id: SettingId): void;
  onNextSetting(): void;
  onToggleSpin(): void;
}

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing #${id} in index.html`);
  return el as T;
}

/** Wires the DOM controls (material picker, shape and setting toggles, keyboard) and owns the text overlays. */
export class Controls {
  readonly top = byId('top');
  readonly dock = byId('dock');
  private readonly hint = byId('hint');
  private readonly readoutEl = byId('readout');
  private readonly fpsEl = byId('fps');
  private readonly spinButton = byId<HTMLButtonElement>('spin');
  private readonly materialButtons = [...document.querySelectorAll<HTMLButtonElement>('.picker button')];
  private readonly shapeButtons = [...document.querySelectorAll<HTMLButtonElement>('.shapes button')];
  private readonly settingButtons = [...document.querySelectorAll<HTMLButtonElement>('.settings button')];
  private readonly themeColor = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  private lastReadout = '';
  private fpsFrames = 0;
  private fpsSince = 0;

  constructor(handlers: ControlHandlers) {
    radioGroup(this.materialButtons, (b) => handlers.onMaterial(b.dataset['mat'] as MaterialId));
    radioGroup(this.shapeButtons, (b) => handlers.onShape(b.dataset['shape'] as ShapeId));
    radioGroup(this.settingButtons, (b) => handlers.onSetting(b.dataset['setting'] as SettingId));
    this.spinButton.addEventListener('click', () => handlers.onToggleSpin());

    window.addEventListener('keydown', (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const material = MATERIAL_KEYS[e.key];
      if (material) handlers.onMaterial(material);
      const shape = SHAPE_KEYS[e.key.toLowerCase()];
      if (shape) handlers.onShape(shape);
      if (e.key.toLowerCase() === 'b') handlers.onNextSetting();
      if (e.key.toLowerCase() === 's') handlers.onToggleSpin();
    });
  }

  showMaterial(id: MaterialId, hint: string): void {
    document.body.dataset['mat'] = id;
    this.hint.textContent = hint;
    for (const b of this.materialButtons) b.setAttribute('aria-checked', String(b.dataset['mat'] === id));
  }

  showShape(id: ShapeId): void {
    document.body.dataset['shape'] = id;
    for (const b of this.shapeButtons) b.setAttribute('aria-checked', String(b.dataset['shape'] === id));
  }

  showSetting(id: SettingId): void {
    document.body.dataset['setting'] = id;
    this.themeColor?.setAttribute('content', THEME_COLORS[id]);
    for (const b of this.settingButtons) b.setAttribute('aria-checked', String(b.dataset['setting'] === id));
  }

  showAutoSpin(on: boolean): void {
    this.spinButton.setAttribute('aria-pressed', String(on));
    this.spinButton.title = on ? 'Stop spinning' : 'Start spinning';
  }

  /** Call once per frame; refreshes the fps counter twice a second. */
  countFrame(now: number): void {
    this.fpsFrames++;
    if (this.fpsSince === 0) this.fpsSince = now;
    const elapsed = now - this.fpsSince;
    if (elapsed < 500) return;
    this.fpsEl.textContent = `${Math.round(((this.fpsFrames - 1) * 1000) / elapsed)} fps`;
    this.fpsFrames = 1;
    this.fpsSince = now;
  }

  /** Cheap to call every frame: only touches the DOM when the text changes. */
  showReadout([title, line1, line2]: Readout): void {
    const key = `${title}|${line1}|${line2}`;
    if (key === this.lastReadout) return;
    this.lastReadout = key;
    const b = document.createElement('b');
    b.textContent = title;
    this.readoutEl.replaceChildren(b, document.createElement('br'), `${line1} · ${line2}`);
  }
}

/** Click to select, arrow keys to move between options, like a native radio group. */
function radioGroup(buttons: HTMLButtonElement[], select: (b: HTMLButtonElement) => void): void {
  buttons.forEach((button, i) => {
    button.addEventListener('click', () => select(button));
    button.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      const step = e.key === 'ArrowRight' ? 1 : buttons.length - 1;
      const next = buttons[(i + step) % buttons.length]!;
      next.focus();
      select(next);
    });
  });
}
