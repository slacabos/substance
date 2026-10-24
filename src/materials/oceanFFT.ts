import { HalfFloatType, LinearFilter, LinearMipmapLinearFilter, RepeatWrapping } from 'three';
import { StorageTexture, type ComputeNode, type StorageBufferNode, type WebGPURenderer } from 'three/webgpu';
import {
  atan,
  cos,
  exp,
  float,
  Fn,
  instancedArray,
  instanceIndex,
  max,
  select,
  sin,
  sqrt,
  textureStore,
  uint,
  uniform,
  uvec2,
  vec4,
} from 'three/tsl';
import { createOceanSpectrum, OCEAN_REPEAT, type OceanCascade, type OceanSpectrum } from '../core/oceanFFT';
import { OCEAN_DEFAULTS, type OceanTuning } from '../core/oceanTuning';
import { oceanWind, seaGain, WIND } from '../core/oceanWind';

type Vec4Buffer = StorageBufferNode<'vec4'>;

/** How long whitecaps linger after a crest stops breaking (seconds, to 1/e). */
const FOAM_LIFE = 2.2;

/**
 * The FFT ocean on the GPU (Tessendorf). Each frame turns every wave's gain toward the live wind
 * (core/oceanWind.ts: short waves follow it within seconds, swell for a minute), advances every cascade's
 * spectrum, runs a 2D inverse
 * Stockham FFT as one compute pass per stage, and writes two tiling textures per cascade:
 * displacement (x, height, z, dDx/dz) and derivatives (dh/dx, dh/dz, dDx/dx, dDz/dz), in the tile's own axes.
 * A third texture remembers foam: how far the surface has bunched up lately (1 - Jacobian), held at its peak
 * and fading over FOAM_LIFE. Tile space follows the water, so foam stays put while waves pass beneath it.
 * Mipmaps are rebuilt after every write, so far-away waves filter rather than shimmer.
 */
export class OceanFFT {
  readonly size: number;
  readonly cascades: readonly OceanCascade[];
  readonly displacement: StorageTexture[];
  readonly derivatives: StorageTexture[];
  /** x: recent folding of the surface at the tuned amplitude, the start of whitecaps. */
  readonly foam: StorageTexture[];
  /** Slope variance each cascade carries at amplitude 1; what the mips average away becomes roughness. */
  readonly slopeVariance: readonly number[];

  private readonly renderer: WebGPURenderer;
  private readonly time = uniform(0);
  private readonly dt = uniform(0);
  /**
   * Every wave's amplitude for the live wind against the default one the spectrum is built for; what the waves
   * are relaxing toward, so slope statistics can follow it.
   */
  readonly windGain = uniform(1);
  /** Where the wind blows, as OCEAN_WIND_ANGLE. */
  private readonly windAngle = uniform(0);
  /** Read every frame: its wind speed is the mean the live wind wanders around, and the wind control sets it. */
  private readonly tuning: Readonly<OceanTuning>;
  private readonly amplitude: number;
  private readonly choppiness: number;
  private readonly passes: ComputeNode[];
  private lastTime = Number.NaN;

  constructor(renderer: WebGPURenderer, coarse: boolean, tuning: Readonly<OceanTuning>) {
    this.renderer = renderer;
    // built for the default wind and scaled live, so the wind can change (even from calm) without a rebuild
    const spectrum = createOceanSpectrum(coarse, { ...tuning, windSpeed: OCEAN_DEFAULTS.windSpeed });
    this.size = spectrum.size;
    this.cascades = spectrum.cascades;
    this.slopeVariance = spectrum.slopeVariance;
    this.tuning = tuning;
    this.amplitude = tuning.amplitude;
    this.choppiness = tuning.choppiness * 0.5;
    const texture = () => {
      const t = new StorageTexture(this.size, this.size);
      t.type = HalfFloatType;
      t.wrapS = t.wrapT = RepeatWrapping;
      t.magFilter = LinearFilter;
      t.minFilter = LinearMipmapLinearFilter;
      t.generateMipmaps = true;
      return t;
    };
    this.displacement = this.cascades.map(texture);
    this.derivatives = this.cascades.map(texture);
    this.foam = this.cascades.map(texture);
    this.passes = this.createPasses(spectrum);
  }

  /** Advances the waves to `time` (seconds); repeated calls for the same time do nothing. */
  update(time: number): void {
    if (time === this.lastTime) return;
    this.dt.value = Number.isFinite(this.lastTime) ? Math.min(0.1, Math.max(0, time - this.lastTime)) : 0;
    this.lastTime = time;
    this.time.value = time % OCEAN_REPEAT;
    const wind = oceanWind(time, this.tuning.windSpeed);
    this.windAngle.value = wind.angle;
    this.windGain.value = seaGain(wind.speed);
    this.renderer.compute(this.passes);
  }

  private createPasses(spectrum: OceanSpectrum): ComputeNode[] {
    const n = this.size;
    const lines = n * n;
    const count = this.cascades.length * lines;
    const h0 = this.storage(spectrum.h0);
    const waves = this.storage(spectrum.waves);
    const shape = this.storage(spectrum.shape);
    const gain = instancedArray(spectrum.gain, 'float');
    // Two complex fields per vec4, [F0, F1] and [F2, F3] (see evolveMode), ping-ponged through the stages.
    const sets: [Vec4Buffer, Vec4Buffer][] = [
      [this.storage(count), this.storage(count)],
      [this.storage(count), this.storage(count)],
    ];

    // relaxGain (core/oceanWind.ts) toward the Donelan–Banner spread toward this wave, for the live wind
    const turn = Fn(() => {
      const i = instanceIndex;
      const form = shape.element(i);
      const off = form.z.sub(this.windAngle);
      const beta = form.x.mul(atan(sin(off), cos(off)));
      const cosh = exp(beta).add(exp(beta.negate())).mul(0.5);
      const target = sqrt(form.y).div(cosh).mul(this.windGain);
      const current = gain.element(i);
      const periods = select(target.greaterThan(current), float(WIND.growPeriods), float(WIND.decayPeriods));
      const keep = exp(
        this.dt
          .mul(form.w)
          .div(periods.mul(2 * Math.PI))
          .negate(),
      );
      current.assign(target.add(current.sub(target).mul(keep)));
    })().compute(count);

    const evolve = Fn(() => {
      const i = instanceIndex;
      const w = waves.element(i);
      // each side of the mode with its own gain: this wave's, and its mirror's at -k (modeAmplitude)
      const local = i.mod(lines);
      const x = local.mod(n);
      const y = local.div(n);
      const mirror = i.sub(local).add(uint(n).sub(y).mod(n).mul(n)).add(uint(n).sub(x).mod(n));
      const a = h0
        .element(i)
        .mul(vec4(gain.element(i), gain.element(i), gain.element(mirror), gain.element(mirror)))
        .toVar();
      const phase = w.z.mul(this.time);
      const c = cos(phase).toVar();
      const s = sin(phase).negate().toVar();
      // h = h0 e^{-i w t} + conj(h0(-k)) e^{i w t}: each wave runs downwind (evolveMode)
      const re = a.x.mul(c).sub(a.y.mul(s)).add(a.z.mul(c)).add(a.w.mul(s)).toVar();
      const im = a.x.mul(s).add(a.y.mul(c)).sub(a.z.mul(s)).add(a.w.mul(c)).toVar();
      const kx = w.x;
      const kz = w.y;
      const ux = kx.mul(w.w).toVar();
      const uz = kz.mul(w.w).toVar();
      sets[0]![0]
        .element(i)
        .assign(
          vec4(
            ux.mul(im).sub(im),
            ux.negate().mul(re).add(re),
            uz.mul(im).sub(kx.mul(uz).mul(im)),
            uz.negate().mul(re).add(kx.mul(uz).mul(re)),
          ),
        );
      sets[0]![1]
        .element(i)
        .assign(
          vec4(
            kx.negate().mul(im).sub(kz.mul(re)),
            kx.mul(re).sub(kz.mul(im)),
            kx.mul(ux).mul(re).sub(kz.mul(uz).mul(im)),
            kx.mul(ux).mul(im).add(kz.mul(uz).mul(re)),
          ),
        );
    })().compute(count);

    // stockhamStage (core/oceanFFT.ts) along rows, then columns; each thread does one butterfly in all fields
    const half = n / 2;
    const stage = (vertical: boolean, p: number, from: number) =>
      Fn(() => {
        const j = instanceIndex;
        const base = j.div(lines / 2).mul(lines);
        const rem = j.mod(lines / 2);
        const line = rem.div(half);
        const t = rem.mod(half);
        const k = t.bitAnd(p - 1);
        const at = (q: typeof t) => (vertical ? base.add(q.mul(n)).add(line) : base.add(line.mul(n)).add(q));
        const angle = float(k).mul(Math.PI / p);
        const c = cos(angle).toVar();
        const s = sin(angle).toVar();
        const o = t.mul(2).sub(k);
        for (let f = 0; f < 2; f++) {
          const src = sets[from]![f]!;
          const dst = sets[1 - from]![f]!;
          const u0 = src.element(at(t)).toVar();
          const x = src.element(at(t.add(half))).toVar();
          const u1 = vec4(
            x.x.mul(c).sub(x.y.mul(s)),
            x.x.mul(s).add(x.y.mul(c)),
            x.z.mul(c).sub(x.w.mul(s)),
            x.z.mul(s).add(x.w.mul(c)),
          ).toVar();
          dst.element(at(o)).assign(u0.add(u1));
          dst.element(at(o.add(p))).assign(u0.sub(u1));
        }
      })().compute(count / 2);
    const stages: ComputeNode[] = [];
    let from = 0;
    for (const vertical of [false, true]) {
      for (let p = 1; p < n; p *= 2) {
        stages.push(stage(vertical, p, from));
        from = 1 - from;
      }
    }

    const [displacement, derivatives] = sets[from]!;
    const history = instancedArray(count, 'float');
    const stretch = this.amplitude * this.choppiness;
    const assemble = this.cascades.map((_, c) =>
      Fn(() => {
        const i = instanceIndex;
        const at = i.add(c * lines);
        const texel = uvec2(i.mod(n), i.div(n));
        const disp = displacement.element(at).toVar();
        const deriv = derivatives.element(at).toVar();
        textureStore(this.displacement[c]!, texel, disp);
        textureStore(this.derivatives[c]!, texel, deriv);
        // J - 1 for the horizontal displacement x + stretch * D
        const xx = deriv.z.mul(stretch);
        const zz = deriv.w.mul(stretch);
        const xz = disp.w.mul(stretch);
        const folding = max(xx.add(zz).add(xx.mul(zz)).sub(xz.mul(xz)).negate(), 0);
        const memory = max(history.element(at).mul(exp(this.dt.mul(-1 / FOAM_LIFE))), folding);
        history.element(at).assign(memory);
        textureStore(this.foam[c]!, texel, vec4(memory, 0, 0, 1));
      })().compute(lines),
    );
    return [turn, evolve, ...stages, ...assemble];
  }

  private storage(data: Float32Array | number): Vec4Buffer {
    return instancedArray(data, 'vec4') as Vec4Buffer;
  }
}
