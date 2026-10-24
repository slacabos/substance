/**
 * The hand of god keeps time. Taps that land on a steady beat push harder the longer the beat holds, so a few
 * well-timed taps raise a bigger wave; an off-beat tap starts over. (The water helps too: pushes that land
 * before it has recovered stack up on their own.)
 */
export class TapRhythm {
  /** Gaps between taps that can count as a beat, seconds. */
  static readonly SLOWEST = 1.6;
  static readonly FASTEST = 0.25;
  /** How far a gap may stray from the beat, as a share of it. */
  static readonly TOLERANCE = 0.3;
  static readonly MAX_STREAK = 4;
  /** Extra push per tap of the streak. */
  static readonly GAIN = 0.3;

  private last = -Infinity;
  private beat = 0;
  private streak = 0;

  /** Records a tap at time t (seconds) and returns how much harder it pushes (1 off the beat). */
  tap(t: number): number {
    const gap = t - this.last;
    this.last = t;
    if (gap < TapRhythm.FASTEST || gap > TapRhythm.SLOWEST) {
      this.beat = 0;
      this.streak = 0;
      return 1;
    }
    const onBeat = this.beat > 0 && Math.abs(gap - this.beat) <= this.beat * TapRhythm.TOLERANCE;
    this.streak = onBeat ? Math.min(TapRhythm.MAX_STREAK, this.streak + 1) : 0;
    // the beat follows the player's tempo as it drifts
    this.beat = onBeat ? this.beat * 0.6 + gap * 0.4 : gap;
    return 1 + TapRhythm.GAIN * this.streak;
  }

  reset(): void {
    this.last = -Infinity;
    this.beat = 0;
    this.streak = 0;
  }
}
