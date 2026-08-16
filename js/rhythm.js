// Rhythm analysis: energy-based beat tracking (Ellis-style dynamic
// programming over an onset envelope, following the singer's tempo drift),
// tempo octave-folding, and the note-anchored grid that classifies each
// inter-onset interval as a musical duration so a wandering internal tempo
// never accumulates timing error.

import { ANALYSIS_SR, HOP, median } from "./dsp.js";

// Musical durations (in beats) an interval between sung onsets may mean,
// with a prior: simple values (whole beats, then half-beats) are far more
// common in sung melodies, so off-grid readings like 1.5 or 2.5 need
// stronger timing evidence to win ("the next note, even early, is the
// start of the new bar").
const IOI_DURATIONS = [
  { d: 0.25, penalty: 0.1 },
  { d: 0.5, penalty: 0.02 },
  { d: 0.75, penalty: 0.08 },
  { d: 1, penalty: 0 },
  { d: 1.5, penalty: 0.06 },
  { d: 2, penalty: 0 },
  { d: 2.5, penalty: 0.08 },
  { d: 3, penalty: 0.02 },
  { d: 3.5, penalty: 0.08 },
  { d: 4, penalty: 0 },
  { d: 5, penalty: 0.04 },
  { d: 6, penalty: 0.02 },
  { d: 7, penalty: 0.04 },
  { d: 8, penalty: 0.02 },
  { d: 10, penalty: 0.04 },
  { d: 12, penalty: 0.04 },
  { d: 16, penalty: 0.04 },
];

/**
 * Builds a beat grid anchored to the notes themselves. Each inter-onset
 * interval is classified as the musical duration that best fits the
 * current local tempo ("that was a half note"), the metrical position
 * advances by that amount, and the grid's *time* re-anchors at the next
 * onset — so even a drifting internal tempo never accumulates error. The
 * local tempo itself adapts smoothly (clamped per step) to follow the
 * singer.
 */
export function noteAnchoredGrid(notes, bpmInit, duration) {
  let b = 60 / bpmInit; // current local beat duration
  let pos = 0; // metrical position in beats
  const anchors = [{ pos: 0, t: notes[0].start }];
  for (let i = 0; i + 1 < notes.length; i++) {
    const ioi = notes[i + 1].start - notes[i].start;
    let bestD = 1;
    let bestCost = Infinity;
    for (const { d, penalty } of IOI_DURATIONS) {
      // Timing fit + simplicity prior + a nudge toward landing on a beat.
      const frac = (pos + d) % 1;
      const gridPenalty = frac === 0 ? 0 : frac === 0.5 ? 0.03 : 0.07;
      const cost = Math.abs(Math.log(ioi / (d * b))) + penalty + gridPenalty;
      if (cost < bestCost) { bestCost = cost; bestD = d; }
    }
    // Follow the implied tempo halfway, clamped to ±18% per note.
    const implied = ioi / bestD;
    b = Math.max(b * 0.82, Math.min(b * 1.18, b + 0.5 * (implied - b)));
    pos += bestD;
    anchors.push({ pos, t: notes[i + 1].start });
  }

  // Beat times at integer positions: linear between anchors, extrapolated
  // at the ends with the local beat duration.
  const beats = [];
  const maxPos = Math.ceil(anchors[anchors.length - 1].pos) + 12;
  let ai = 0;
  for (let k = 0; k <= maxPos; k++) {
    while (ai + 1 < anchors.length && anchors[ai + 1].pos <= k) ai++;
    let t;
    if (k <= anchors[0].pos) {
      t = anchors[0].t - (anchors[0].pos - k) * (60 / bpmInit);
    } else if (ai + 1 < anchors.length) {
      const a = anchors[ai];
      const c = anchors[ai + 1];
      t = a.t + ((k - a.pos) / (c.pos - a.pos)) * (c.t - a.t);
    } else {
      const a = anchors[anchors.length - 1];
      t = a.t + (k - a.pos) * b;
    }
    beats.push(t);
    if (t > duration + 5 * b) break;
  }
  return beats;
}

/**
 * Octave-fold a tempo into a plausible vocal range (70-140 BPM). Beat trackers
 * routinely lock onto the eighth-note subdivision on busy/syllabic singing,
 * reporting ~2x; this pulls it back. When given the beat grid it was derived
 * from, the grid is thinned by the same factor so the two stay consistent —
 * important because the arranger groups every 4 beats into a 4/4 bar, so an
 * un-thinned (eighth-spaced) grid would make every "bar" a half-bar and place
 * chords at double rate.
 */
/**
 * Picks the tempo whose metrical grid best explains the sung onsets.
 *
 * The previous rule — take the median spacing of the note-anchored grid and
 * fold it into range — answers "what is the commonest gap?", which is not the
 * same question as "what pulse are these notes on". Measurement showed the
 * median rule lands on a tempo that fits the onsets no better than one
 * derived from perfect transcription, i.e. the estimator, not the note list,
 * was the limit.
 *
 * Each candidate is scored by how many onsets fall near an eighth-note
 * position, MINUS the hit rate expected by chance at that tempo. Without that
 * correction faster grids win automatically (denser grids catch more onsets),
 * which is exactly the octave error a naive fit makes.
 *
 * @returns {number} BPM within [lo, hi]
 */
export function bestGridTempo(notes, { lo = 70, hi = 140, tol = 0.06, phases = 16 } = {}) {
  const onsets = notes.map((n) => n.start);
  if (onsets.length < 4) return 100;
  let bestBpm = 100, bestScore = -Infinity;
  for (let bpm = lo; bpm <= hi; bpm += 0.5) {
    const step = 30 / bpm; // eighth-note period
    let hits = 0;
    for (let p = 0; p < phases; p++) {
      const phase = (p / phases) * step;
      let h = 0;
      for (const t of onsets) {
        const u = (t - phase) / step;
        if (Math.abs(u - Math.round(u)) * step <= tol) h++;
      }
      if (h > hits) hits = h;
    }
    // Chance = the share of the timeline within tol of any grid point.
    const score = hits / onsets.length - Math.min(1, (2 * tol) / step);
    if (score > bestScore) { bestScore = score; bestBpm = bpm; }
  }
  return bestBpm;
}

export function foldTempo(bpm, beats = null) {
  let fold = 1;
  while (bpm > 140) { bpm /= 2; fold *= 2; }
  while (bpm < 70) { bpm *= 2; fold /= 2; }
  if (fold >= 2 && beats) beats = beats.filter((_, i) => i % fold === 0);
  return { tempo: Math.round(bpm), beats };
}

/**
 * Beat tracking. Builds an onset-strength envelope, estimates the global
 * beat period by autocorrelation, then runs Ellis-style dynamic programming
 * to find a beat *sequence* — each onset rewards landing a beat there, and
 * intervals are pulled toward the period by a squared-log penalty. The grid
 * follows the singer's tempo drift instead of assuming a fixed BPM.
 */
export function detectBeats(x, notes, duration) {
  const frameDur = HOP / ANALYSIS_SR;
  const nFrames = Math.max(2, Math.floor(x.length / HOP) - 1);

  // Onset envelope: half-wave-rectified RMS derivative, strongly reinforced
  // by detected note starts (the most reliable vocal onsets).
  const env = new Float32Array(nFrames);
  for (let i = 0; i < nFrames; i++) {
    let sum = 0;
    for (let j = 0; j < HOP; j++) {
      const v = x[i * HOP + j];
      sum += v * v;
    }
    env[i] = Math.sqrt(sum / HOP);
  }
  const onset = new Float32Array(nFrames);
  for (let i = 1; i < nFrames; i++) onset[i] = Math.max(0, env[i] - env[i - 1]);
  let peak = 0;
  for (let i = 0; i < nFrames; i++) peak = Math.max(peak, onset[i]);
  if (peak > 0) for (let i = 0; i < nFrames; i++) onset[i] /= peak;
  for (const n of notes) {
    const idx = Math.round(n.start / frameDur);
    if (idx >= 0 && idx < nFrames) onset[idx] = Math.min(1.5, onset[idx] + 0.6);
  }

  const bpmEstimate = estimatePeriod(onset, frameDur);
  const period = (60 / bpmEstimate) / frameDur; // frames per beat

  // DP: score[t] = onset[t] + best continuation from a previous beat.
  // A chain may also start fresh at any frame (its own onset value).
  const alpha = 1.5;
  const pMin = Math.max(2, Math.round(period * 0.5));
  const pMax = Math.round(period * 2);
  const score = new Float32Array(nFrames);
  const from = new Int32Array(nFrames).fill(-1);
  for (let t = 0; t < nFrames; t++) {
    score[t] = onset[t];
    const lo = Math.max(0, t - pMax);
    for (let p = lo; p <= t - pMin; p++) {
      const pen = -alpha * Math.pow(Math.log((t - p) / period), 2);
      const s = score[p] + pen + onset[t];
      if (s > score[t]) { score[t] = s; from[t] = p; }
    }
  }

  let best = 0;
  for (let t = 1; t < nFrames; t++) if (score[t] > score[best]) best = t;
  const beatFrames = [];
  for (let t = best; t >= 0; t = from[t]) {
    beatFrames.push(t);
    if (from[t] === -1) break;
  }
  beatFrames.reverse();

  let beats = beatFrames.map((f) => f * frameDur);
  if (beats.length < 6) {
    // Take too short/sparse for tracking: fall back to a uniform grid.
    const interval = 60 / bpmEstimate;
    beats = [];
    for (let t = notes[0].start; t < duration + 5 * interval; t += interval) beats.push(t);
    return { tempo: bpmEstimate, beats };
  }

  // Median interval = display tempo; extend the grid past the end of the
  // take so the final bar can complete.
  const intervals = [];
  for (let i = 1; i < beats.length; i++) intervals.push(beats[i] - beats[i - 1]);
  const med = median(intervals);
  let tempo;
  ({ tempo, beats } = foldTempo(60 / med, beats));
  const interval = 60 / tempo;
  while (beats[beats.length - 1] < duration + 5 * interval) {
    beats.push(beats[beats.length - 1] + interval);
  }
  return { tempo, beats };
}

function estimatePeriod(onset, frameDur) {
  const nFrames = onset.length;
  const lagMin = Math.round((60 / 180) / frameDur);
  const lagMax = Math.round((60 / 60) / frameDur);
  let bestLag = -1, bestScore = -Infinity;
  for (let lag = lagMin; lag <= lagMax && lag < nFrames; lag++) {
    let sum = 0;
    for (let i = 0; i + lag < nFrames; i++) sum += onset[i] * onset[i + lag];
    const bpm = 60 / (lag * frameDur);
    const prior = Math.exp(-Math.pow(Math.log2(bpm / 100), 2) / 0.8);
    const s = sum * prior;
    if (s > bestScore) { bestScore = s; bestLag = lag; }
  }
  let bpm = bestLag > 0 ? 60 / (bestLag * frameDur) : 90;
  while (bpm < 70) bpm *= 2;
  while (bpm > 140) bpm /= 2;
  return Math.round(bpm);
}
