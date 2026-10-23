import type { MaterialId, Readout, ShapeId } from '../materials/types';

const MATERIAL_KEYS: Record<string, MaterialId> = { '1': 'water', '2': 'ocean', '3': 'slime', '4': 'fur' };
const SHAPE_KEYS: Record<string, ShapeId> = { o: 'orb', c: 'cube' };

export interface ControlHandlers {
  onMaterial(id: MaterialId): void;
  onShape(id: ShapeId): void;
}

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing #${id} in index.html`);
  return el as T;
}

/** Wires the DOM controls (material picker, shape toggle, keyboard) and owns the text overlays. */
export class Controls {
  readonly top = byId('top');
  readonly dock = byId('dock');
  private readonly hint = byId('hint');
  private readonly readoutEl = byId('readout');
  private readonly materialButtons = [...document.querySelectorAll<HTMLButtonElement>('.picker button')];
  private readonly shapeButtons = [...document.querySelectorAll<HTMLButtonElement>('.shapes button')];
  private lastReadout = '';

  constructor(handlers: ControlHandlers, coarse: boolean) {
    if (!coarse) byId('sub').textContent = 'Drag the background to spin · keys 1–4, O, C';

    radioGroup(this.materialButtons, (b) => handlers.onMaterial(b.dataset['mat'] as MaterialId));
    radioGroup(this.shapeButtons, (b) => handlers.onShape(b.dataset['shape'] as ShapeId));

    window.addEventListener('keydown', (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const material = MATERIAL_KEYS[e.key];
      if (material) handlers.onMaterial(material);
      const shape = SHAPE_KEYS[e.key.toLowerCase()];
      if (shape) handlers.onShape(shape);
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

  /** Cheap to call every frame: only touches the DOM when the text changes. */
  showReadout([title, line1, line2]: Readout): void {
    const key = `${title}|${line1}|${line2}`;
    if (key === this.lastReadout) return;
    this.lastReadout = key;
    const b = document.createElement('b');
    b.textContent = title;
    this.readoutEl.replaceChildren(
      b,
      document.createElement('br'),
      line1,
      document.createElement('br'),
      line2,
    );
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
