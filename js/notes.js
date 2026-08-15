// Note-list post-processing: every transform that takes the raw note list
// and cleans, validates, or reshapes it. Grouped here (rather than split
// across the transcriber and the analyzer) so the full note-hygiene pipeline
// reads top to bottom in one place.
//
// Two families:
//   - cleanNotes(): model-output hygiene that runs inside transcription
//     (amplitude gate, octave ghosts, fragment merge, monophony, outliers).
//   - the exported passes below, applied by analyzeVocal in sequence:
//     refineNote (YIN cross-validation), snapNoteOnsets, applySyllableLock,
//     applyMinNote, simplifyNotes (auto-tune).

import {
  ANALYSIS_SR, FRAME_SIZE, F_MIN, F_MAX, winRms, median, pitchClass, createYin, yinFrame,
  fft, hann, midiToHz,
} from "./dsp.js";

// ---------------------------------------------------------------------------
// Model-output hygiene (runs inside transcribeBasicPitch)
// ---------------------------------------------------------------------------

/**
 * Note hygiene for a solo vocal: the model's spurious output (breath blips,
 * harmonic ghosts, vibrato splits, scoops) is filtered with heuristics that
 * exploit what we know — the source is one voice singing one line.
 */
export function cleanNotes(notes, gateRatio = 0.38) {
  let n = amplitudeGate(notes, gateRatio);
  n = dropOctaveGhosts(n);
  // Merge before monophony: its minimum-length filter would otherwise delete
  // scoop grace notes before they can be absorbed into their main note.
  n = mergeFragments(n);
  n = enforceMonophony(n);
  n = dropOutliers(n);
  return n;
}

/** Spurious notes are far quieter than real singing: gate on the take's median. */
function amplitudeGate(notes, gateRatio) {
  if (notes.length < 4) return notes;
  const med = median(notes.map((n) => n.amp ?? 0));
  return notes.filter((n) => (n.amp ?? med) >= gateRatio * med);
}

/** Faint parallel notes an octave/twelfth/two octaves from a louder note. */
function dropOctaveGhosts(notes) {
  const dead = new Set();
  for (let i = 0; i < notes.length; i++) {
    for (let j = i + 1; j < notes.length; j++) {
      const a = notes[i], b = notes[j];
      if (b.start - a.end > 0.08) break; // sorted by start: rest are later
      const dp = Math.abs(a.midiFloat - b.midiFloat);
      if (Math.abs(dp - 12) >= 0.6 && Math.abs(dp - 19) >= 0.6 && Math.abs(dp - 24) >= 0.6) continue;
      if ((b.amp ?? 0) < 0.55 * (a.amp ?? 0)) dead.add(j);
      else if ((a.amp ?? 0) < 0.55 * (b.amp ?? 0)) dead.add(i);
    }
  }
  return notes.filter((_, i) => !dead.has(i));
}

/** Rejoin vibrato splits (same pitch, tiny gap) and absorb scoop grace notes. */
function mergeFragments(notes) {
  const out = [];
  for (const n of notes) {
    const last = out[out.length - 1];
    // Same-pitch continuation: merge only when the second fragment is much
    // quieter than the first (a vibrato/decay split). The model deliberately
    // splits genuine re-struck notes at the new onset with zero gap and
    // similar amplitude — those must survive as separate notes.
    if (last && n.start - last.end <= 0.03 && Math.abs(n.midiFloat - last.midiFloat) <= 0.6 &&
        (n.amp ?? 1) < 0.75 * (last.amp ?? 1)) {
      last.end = Math.max(last.end, n.end);
      last.dur = last.end - last.start;
      last.amp = Math.max(last.amp ?? 0, n.amp ?? 0);
      continue;
    }
    if (last && last.dur < 0.1 && n.start - last.end <= 0.05 &&
        Math.abs(n.midiFloat - last.midiFloat) <= 2.5 && n.dur > 2 * last.dur) {
      out.pop(); // scoop onset: the long note actually started at the scoop
      n.start = last.start;
      n.dur = n.end - n.start;
    }
    out.push(n);
  }
  return out;
}

/** A short, quiet note more than an octave from both neighbors isn't singing. */
function dropOutliers(notes) {
  if (notes.length < 3) return notes;
  const medAmp = median(notes.map((n) => n.amp ?? 0));
  return notes.filter((n, i) => {
    if (n.dur >= 0.15 || (n.amp ?? medAmp) >= medAmp) return true;
    const dPrev = i > 0 ? Math.abs(n.midiFloat - notes[i - 1].midiFloat) : Infinity;
    const dNext = i < notes.length - 1 ? Math.abs(n.midiFloat - notes[i + 1].midiFloat) : Infinity;
    return Math.min(dPrev, dNext) <= 12;
  });
}

/**
 * Basic Pitch is polyphonic; a vocal isn't. Where notes overlap heavily
 * (usually a harmonic ghost), keep the louder one; small overlaps just trim
 * the earlier note.
 */
function enforceMonophony(notes) {
  const out = [];
  for (const n of notes) {
    const last = out[out.length - 1];
    if (last && n.start < last.end - 0.02) {
      const overlap = Math.min(last.end, n.end) - n.start;
      if (overlap > 0.5 * Math.min(last.dur, n.dur)) {
        if ((n.amp ?? 0) > (last.amp ?? 0)) out[out.length - 1] = n;
        continue;
      }
      last.end = n.start;
      last.dur = last.end - last.start;
    }
    out.push(n);
  }
  return out.filter((n) => n.dur >= 0.08);
}

// ---------------------------------------------------------------------------
// YIN cross-validation
// ---------------------------------------------------------------------------

/**
 * Cross-validates a transcribed note against the raw signal with YIN probes
 * spread across its interior. Returns false (drop) when the audio shows no
 * periodicity — a breath or consonant, not singing. When the probes agree
 * with each other, their median refines the note's exact pitch, including
 * correcting harmonic/octave confusions (±12/19/24 st).
 *
 * Pass a shared createYin() context when validating many notes so the
 * working buffers are allocated once per take, not once per note.
 */
export function refineNote(x, note, yin = createYin()) {
  const t0 = note.start + 0.03;
  const t1 = note.end - 0.03;
  if (t1 <= t0) return true; // too short to judge fairly
  const nProbes = 9;
  let probed = 0;
  const midis = [];
  for (let i = 0; i < nProbes; i++) {
    const t = t0 + (i / (nProbes - 1)) * (t1 - t0);
    const off = Math.round(t * ANALYSIS_SR);
    if (off < 0 || off + FRAME_SIZE >= x.length) continue;
    probed++;
    const f = yinFrame(x, off, yin);
    if (f.voiced) midis.push(f.midi);
  }
  if (probed === 0) return true;
  if (midis.length / probed < 0.34) return false; // no periodic pitch: not a sung note

  midis.sort((a, b) => a - b);
  const m = midis[Math.floor(midis.length / 2)];
  const cluster = midis.filter((v) => Math.abs(v - m) <= 0.5).length;
  if (cluster >= Math.max(2, Math.ceil(midis.length / 2))) {
    const dist = Math.abs(m - note.midiFloat);
    if (dist < 0.8 || Math.abs(dist - 12) < 0.8 || Math.abs(dist - 19) < 0.8 || Math.abs(dist - 24) < 0.8) {
      note.midiFloat = m; // refine in place, or snap the octave to the signal
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Held-note consolidation
// ---------------------------------------------------------------------------

/**
 * Rejoins held notes the model split into same-pitch fragments, using the
 * audio as the referee. cleanNotes' amplitude heuristic can't tell a spurious
 * split of one held note from a genuine re-articulation ("doo doo" on one
 * pitch) — but the signal can: re-articulating closes the vocal tract, so
 * energy dips at the boundary; a held note keeps sounding straight through.
 * Merges adjacent notes within `pitchTol` semitones and `maxGap` seconds when
 * the quietest 10 ms window across the junction retains at least `dipRatio`
 * of the fragments' own interior level.
 *
 * @returns {{notes: Array, merged: number}}
 */
export function consolidateRepeats(x, notes, { maxGap = 0.12, pitchTol = 0.6, dipRatio = 0.5 } = {}) {
  const interior = (n) => {
    const probes = [0.3, 0.5, 0.7].map((f) => {
      const t = n.start + f * (n.end - n.start);
      return winRms(x, t - 0.005, t + 0.005);
    });
    return median(probes);
  };
  const out = [];
  let merged = 0;
  for (const n of notes) {
    const last = out[out.length - 1];
    if (last && n.start - last.end <= maxGap && n.start >= last.end - 0.001 &&
        Math.abs(n.midiFloat - last.midiFloat) <= pitchTol) {
      const level = Math.min(interior(last), interior(n));
      // Quietest 10 ms window across the junction (a little into each note).
      let dip = Infinity;
      const t0 = Math.max(last.start, last.end - 0.04);
      const t1 = Math.min(n.end, n.start + 0.04);
      for (let t = t0; t + 0.01 <= t1 + 1e-6; t += 0.005) {
        dip = Math.min(dip, winRms(x, t, t + 0.01));
      }
      if (level > 0 && dip < Infinity && dip >= dipRatio * level) {
        const wa = last.end - last.start, wb = n.end - n.start;
        last.midiFloat = (last.midiFloat * wa + n.midiFloat * wb) / (wa + wb);
        last.end = Math.max(last.end, n.end);
        last.dur = last.end - last.start;
        last.amp = Math.max(last.amp ?? 0, n.amp ?? 0);
        merged++;
        continue;
      }
    }
    out.push(n);
  }
  return { notes: out, merged };
}

// ---------------------------------------------------------------------------
// Octave correction (sub-harmonic summation)
// ---------------------------------------------------------------------------

const SHS_FFT = 2048; // 128 ms @ 16 kHz; 7.8 Hz bins resolve octaves easily
const SHS_HARMONICS = 10;
const SHS_DECAY = 0.84; // weight of the k-th harmonic (Hermes SHS)
const SHS_MARGIN = 1.15; // an octave alternative must beat the reported salience by this

/**
 * Corrects octave errors that YIN and the neural model share. At each note's
 * stable centre it takes a spectrum and scores the reported pitch against its
 * octave neighbours by sub-harmonic summation — the true fundamental wins
 * because *all* its harmonics contribute, while an octave-too-high pick only
 * lands on the even ones. Only octave shifts (chroma-preserving) are allowed,
 * and only when the alternative is clearly more salient, so it fixes the
 * weak-fundamental (low-voice) failure without inventing new pitch classes.
 *
 * @returns {number} how many notes were octave-shifted
 */
export function octaveCorrect(x, notes) {
  const half = SHS_FFT / 2;
  const binHz = ANALYSIS_SR / SHS_FFT;
  const win = hann(SHS_FFT);
  const re = new Float64Array(SHS_FFT);
  const im = new Float64Array(SHS_FFT);
  const mag = new Float32Array(half);
  let corrected = 0;

  for (const n of notes) {
    let off = Math.round(((n.start + n.end) / 2) * ANALYSIS_SR) - SHS_FFT / 2;
    off = Math.max(0, Math.min(x.length - SHS_FFT, off));
    if (off < 0) continue; // clip shorter than the window
    for (let i = 0; i < SHS_FFT; i++) { re[i] = x[off + i] * win[i]; im[i] = 0; }
    fft(re, im);
    for (let i = 0; i < half; i++) mag[i] = Math.hypot(re[i], im[i]);

    const base = shs(mag, binHz, midiToHz(n.midiFloat), half);
    let bestShift = 0;
    let bestScore = base * SHS_MARGIN;
    for (const shift of [-24, -12, 12]) {
      const f = midiToHz(n.midiFloat + shift);
      if (f < F_MIN * 0.5 || f > F_MAX * 1.5) continue;
      const s = shs(mag, binHz, f, half);
      if (s > bestScore) { bestScore = s; bestShift = shift; }
    }
    if (bestShift !== 0) { n.midiFloat += bestShift; corrected++; }
  }
  return corrected;
}

/**
 * Measures the real harmonic stack of each melody note from the audio: an
 * FFT at the note's stable centre, then the amplitude of each partial k·f0
 * (peak within ±3% to tolerate vibrato/inharmonicity). Each partial becomes
 * a note at its nearest MIDI pitch with `prob` = amplitude relative to the
 * note's strongest partial — so exported velocity encodes the singer's own
 * timbre. Partials below `floor` of the max are dropped.
 *
 * @returns {Array<{start,end,midi,prob}>} one entry per audible partial
 */
export function harmonicStack(x, notes, { maxHarmonics = 10, floor = 0.05 } = {}) {
  const half = SHS_FFT / 2;
  const binHz = ANALYSIS_SR / SHS_FFT;
  const win = hann(SHS_FFT);
  const re = new Float64Array(SHS_FFT);
  const im = new Float64Array(SHS_FFT);
  const mag = new Float32Array(half);
  const out = [];

  for (const n of notes) {
    let off = Math.round(((n.start + n.end) / 2) * ANALYSIS_SR) - SHS_FFT / 2;
    off = Math.max(0, Math.min(x.length - SHS_FFT, off));
    if (off < 0) continue; // clip shorter than the window
    for (let i = 0; i < SHS_FFT; i++) { re[i] = x[off + i] * win[i]; im[i] = 0; }
    fft(re, im);
    for (let i = 0; i < half; i++) mag[i] = Math.hypot(re[i], im[i]);

    const f0 = midiToHz(n.midiFloat ?? n.midi);
    const partials = [];
    for (let k = 1; k <= maxHarmonics; k++) {
      const fk = k * f0;
      if (fk >= (ANALYSIS_SR / 2) * 0.95) break;
      const lo = Math.max(1, Math.floor((fk * 0.97) / binHz));
      const hi = Math.min(half - 1, Math.ceil((fk * 1.03) / binHz));
      let m = 0;
      for (let b = lo; b <= hi; b++) if (mag[b] > m) m = mag[b];
      partials.push({ k, m });
    }
    let maxM = 1e-9;
    for (const p of partials) if (p.m > maxM) maxM = p.m;
    for (const p of partials) {
      const rel = p.m / maxM;
      if (rel < floor) continue;
      const midiK = Math.round(69 + 12 * Math.log2((p.k * f0) / 440));
      if (midiK < 21 || midiK > 108) continue;
      out.push({ start: n.start, end: n.end, midi: midiK, prob: rel });
    }
  }
  out.sort((a, b) => a.start - b.start || a.midi - b.midi);
  return out;
}

/**
 * The take's average harmonic recipe: the same per-note partial measurement
 * as harmonicStack, aggregated duration-weighted across every note and
 * normalized so the strongest harmonic is 1. Feed the result to
 * AudioContext.createPeriodicWave (as the imag/sine array) to get an
 * oscillator with the singer's own overtone structure — the voice-timbre
 * synth. amps[0] is DC (unused, zero); amps[k] is harmonic k.
 */
export function voiceTimbre(x, notes, { maxHarmonics = 10 } = {}) {
  const half = SHS_FFT / 2;
  const binHz = ANALYSIS_SR / SHS_FFT;
  const win = hann(SHS_FFT);
  const re = new Float64Array(SHS_FFT);
  const im = new Float64Array(SHS_FFT);
  const mag = new Float32Array(half);
  const sums = new Float64Array(maxHarmonics + 1);
  const wts = new Float64Array(maxHarmonics + 1);

  for (const n of notes) {
    let off = Math.round(((n.start + n.end) / 2) * ANALYSIS_SR) - SHS_FFT / 2;
    off = Math.max(0, Math.min(x.length - SHS_FFT, off));
    if (off < 0) continue;
    for (let i = 0; i < SHS_FFT; i++) { re[i] = x[off + i] * win[i]; im[i] = 0; }
    fft(re, im);
    for (let i = 0; i < half; i++) mag[i] = Math.hypot(re[i], im[i]);

    const f0 = midiToHz(n.midiFloat ?? n.midi);
    let noteMax = 1e-9;
    const ms = [];
    for (let k = 1; k <= maxHarmonics; k++) {
      const fk = k * f0;
      if (fk >= (ANALYSIS_SR / 2) * 0.95) break;
      const lo = Math.max(1, Math.floor((fk * 0.97) / binHz));
      const hi = Math.min(half - 1, Math.ceil((fk * 1.03) / binHz));
      let m = 0;
      for (let b = lo; b <= hi; b++) if (mag[b] > m) m = mag[b];
      ms.push(m);
      if (m > noteMax) noteMax = m;
    }
    const w = Math.max(0.05, n.end - n.start);
    ms.forEach((m, i) => { sums[i + 1] += (m / noteMax) * w; wts[i + 1] += w; });
  }

  const amps = new Float32Array(maxHarmonics + 1);
  let max = 1e-9;
  for (let k = 1; k <= maxHarmonics; k++) {
    amps[k] = wts[k] > 0 ? sums[k] / wts[k] : 0;
    if (amps[k] > max) max = amps[k];
  }
  for (let k = 1; k <= maxHarmonics; k++) amps[k] /= max;
  return amps;
}

/** Sub-harmonic summation salience of fundamental f0Hz. */
function shs(mag, binHz, f0Hz, half) {
  let s = 0;
  for (let k = 1; k <= SHS_HARMONICS; k++) {
    const b = (k * f0Hz) / binHz;
    const i0 = Math.floor(b);
    if (i0 + 1 >= half) break;
    const frac = b - i0;
    s += SHS_DECAY ** (k - 1) * (mag[i0] * (1 - frac) + mag[i0 + 1] * frac);
  }
  return s;
}

// ---------------------------------------------------------------------------
// Onset snapping
// ---------------------------------------------------------------------------

const SNAP_HOP = 96; // 6 ms at 16 kHz
const SNAP_WINDOW = 0.07; // seconds searched around each transcribed onset

/**
 * Snaps note starts to nearby energy rises. The transcriber's note
 * boundaries wait for stable pitch, smearing onsets by tens of ms; the
 * energy envelope reacts at the moment of articulation. For each note we
 * look ±SNAP_WINDOW around its start for a significant rise in a fine
 * (6 ms) onset envelope and move the start to the *beginning* of that rise
 * (walking back from the steepest point). Notes with no significant rise
 * nearby — legato transitions, soft entries — are left untouched.
 *
 * @returns {number} how many note starts were adjusted
 */
export function snapNoteOnsets(x, notes) {
  const hopDur = SNAP_HOP / ANALYSIS_SR;
  const win = SNAP_HOP * 2; // 12 ms RMS window
  const nFrames = Math.floor((x.length - win) / SNAP_HOP);
  if (nFrames < 8) return 0;

  const rms = new Float32Array(nFrames);
  for (let i = 0; i < nFrames; i++) {
    let sum = 0;
    const off = i * SNAP_HOP;
    for (let j = 0; j < win; j++) {
      const v = x[off + j];
      sum += v * v;
    }
    rms[i] = Math.sqrt(sum / win);
  }
  const onset = new Float32Array(nFrames);
  for (let i = 1; i < nFrames; i++) onset[i] = Math.max(0, rms[i] - rms[i - 1]);

  // Significance threshold: a real articulation must clear the take's
  // typical frame-to-frame flux by a wide margin.
  const sorted = [...onset].sort((a, b) => a - b);
  const medFlux = sorted[nFrames >> 1];
  const thresh = Math.max(3 * medFlux, 0.04 * sorted[nFrames - 1]);

  let snapped = 0;
  for (let k = 0; k < notes.length; k++) {
    const n = notes[k];
    const lo = Math.max(1, Math.round((n.start - SNAP_WINDOW) / hopDur));
    const hi = Math.min(nFrames - 1, Math.round((n.start + SNAP_WINDOW) / hopDur));
    let peakIdx = -1;
    for (let i = lo; i <= hi; i++) {
      if (onset[i] > thresh && (peakIdx === -1 || onset[i] > onset[peakIdx])) peakIdx = i;
    }
    if (peakIdx === -1) continue; // no articulation nearby: legato, keep as-is

    // Walk back from the steepest point to where the rise begins (max 48 ms).
    let riseIdx = peakIdx;
    while (riseIdx > lo && riseIdx > peakIdx - 8 && onset[riseIdx - 1] >= 0.25 * onset[peakIdx]) riseIdx--;

    const t = riseIdx * hopDur;
    if (t >= n.end - 0.05 || Math.abs(t - n.start) < hopDur) continue;
    const prev = notes[k - 1];
    if (prev && prev.end > t) {
      if (t <= prev.start + 0.05) continue; // would crush the previous note
      prev.end = t;
      prev.dur = prev.end - prev.start;
    }
    n.start = t;
    n.dur = n.end - n.start;
    snapped++;
  }
  return snapped;
}

/**
 * Snaps note ENDS to nearby energy decays — the mirror of snapNoteOnsets, and
 * aimed at the pipeline's weakest metric (offsets score far below onsets).
 * The transcriber ends a note when its pitch estimate destabilises, which
 * drifts late through a release tail or early through a wobble; the energy
 * envelope falls at the moment the sound actually stops.
 *
 * For each note we look ±OFF_WINDOW around its end for the steepest fall in
 * the same 6 ms envelope, then walk FORWARD to where the fall flattens out —
 * the note is over once the decay finishes, not where it began. Notes with no
 * clear decay nearby (legato transitions into the next note, notes that fade)
 * are left untouched.
 *
 * @returns {number} how many note ends were adjusted
 */
export function snapNoteOffsets(x, notes, { window: winSec = 0.09, minDur = 0.06 } = {}) {
  const hopDur = SNAP_HOP / ANALYSIS_SR;
  const win = SNAP_HOP * 2;
  const nFrames = Math.floor((x.length - win) / SNAP_HOP);
  if (nFrames < 8) return 0;

  const rms = new Float32Array(nFrames);
  for (let i = 0; i < nFrames; i++) {
    let sum = 0;
    const off = i * SNAP_HOP;
    for (let j = 0; j < win; j++) {
      const v = x[off + j];
      sum += v * v;
    }
    rms[i] = Math.sqrt(sum / win);
  }
  // Falling flux: the opposite sign to the onset envelope.
  const decay = new Float32Array(nFrames);
  for (let i = 1; i < nFrames; i++) decay[i] = Math.max(0, rms[i - 1] - rms[i]);

  const sorted = [...decay].sort((a, b) => a - b);
  const medFlux = sorted[nFrames >> 1];
  const thresh = Math.max(3 * medFlux, 0.04 * sorted[nFrames - 1]);

  let snapped = 0;
  for (let k = 0; k < notes.length; k++) {
    const n = notes[k];
    const lo = Math.max(1, Math.round((n.end - winSec) / hopDur));
    const hi = Math.min(nFrames - 1, Math.round((n.end + winSec) / hopDur));
    let peakIdx = -1;
    for (let i = lo; i <= hi; i++) {
      if (decay[i] > thresh && (peakIdx === -1 || decay[i] > decay[peakIdx])) peakIdx = i;
    }
    if (peakIdx === -1) continue; // no clear release: legato or fade, leave it

    // Walk forward to where the decay flattens (max 48 ms).
    let fallIdx = peakIdx;
    while (fallIdx < hi && fallIdx < peakIdx + 8 && decay[fallIdx + 1] >= 0.25 * decay[peakIdx]) fallIdx++;

    const t = fallIdx * hopDur;
    if (t - n.start < minDur || Math.abs(t - n.end) < hopDur) continue;
    const next = notes[k + 1];
    if (next && t > next.start) continue; // never swallow the next note
    n.end = t;
    n.dur = n.end - n.start;
    snapped++;
  }
  return snapped;
}

// ---------------------------------------------------------------------------
// Syllable lock
// ---------------------------------------------------------------------------

/**
 * Syllable lock. A stop-consonant syllable like "Doo" stamps every real
 * note with a measurable signature: a closure dip (near-silence right
 * before the onset), a fast release burst, and a consistent vowel timbre.
 * We learn that signature from the take itself (median ± MAD over all
 * notes) — so any consistent syllable works — and engage only when the
 * articulation actually measures consistent. When engaged:
 *  1. Notes deviating from the signature on 2+ features (and not long and
 *     loud) are dropped — breaths and stray sounds don't begin like a "Doo".
 *  2. Same-pitch fragments at a boundary with no closure dip are merged —
 *     a real re-strike has a dip, a model split doesn't. Measured, not
 *     guessed.
 */
export function applySyllableLock(x, notes) {
  const info = { enabled: true, engaged: false, dropped: 0, merged: 0, notes, reason: "" };
  if (notes.length < 8) {
    info.reason = "needs 8+ notes to calibrate";
    return info;
  }

  const feats = notes.map((n) => onsetFeatures(x, n));
  const med = {
    dip: median(feats.map((f) => f.dip)),
    rise: median(feats.map((f) => f.rise)),
    hf: median(feats.map((f) => f.hf)),
  };
  const mad = {
    dip: median(feats.map((f) => Math.abs(f.dip - med.dip))),
    rise: median(feats.map((f) => Math.abs(f.rise - med.rise))),
    hf: median(feats.map((f) => Math.abs(f.hf - med.hf))),
  };

  // Engage only when the take's onsets cluster tightly — otherwise (lyrics,
  // mixed articulation) stand down rather than damage the transcription.
  if (mad.rise > 0.025 || mad.hf > 0.05 || mad.dip > 0.15) {
    info.reason = "onsets too varied for a syllable template";
    return info;
  }
  info.engaged = true;

  const ampMed = median(notes.map((n) => n.amp ?? 1));

  // 1. Drop notes that don't begin like the singer's syllable.
  const kept = [];
  notes.forEach((n, i) => {
    const f = feats[i];
    let deviant = 0;
    if (f.dip > med.dip + Math.max(4 * mad.dip, 0.12)) deviant++;
    if (f.rise > med.rise + Math.max(4 * mad.rise, 0.03)) deviant++;
    if (Math.abs(f.hf - med.hf) > Math.max(4 * mad.hf, 0.06)) deviant++;
    if (deviant >= 2 && (n.dur < 0.3 || (n.amp ?? ampMed) < ampMed)) {
      info.dropped++;
    } else {
      kept.push(n);
    }
  });

  // 2. Same-pitch boundary with no closure dip = model split, not re-strike.
  const out = [];
  for (const n of kept) {
    const last = out[out.length - 1];
    if (last && Math.abs(n.midiFloat - last.midiFloat) <= 0.6 && n.start - last.end <= 0.08) {
      const bDip = boundaryDip(x, n.start);
      if (bDip > med.dip + Math.max(3 * mad.dip, 0.12)) {
        last.end = Math.max(last.end, n.end);
        last.dur = last.end - last.start;
        last.amp = Math.max(last.amp ?? 0, n.amp ?? 0);
        info.merged++;
        continue;
      }
    }
    out.push(n);
  }
  info.notes = out;
  return info;
}

function onsetFeatures(x, n) {
  // Peak of the attack region.
  let peak = 1e-6;
  let tPeak = n.start;
  for (let t = n.start; t <= n.start + 0.08; t += 0.005) {
    const r = winRms(x, t, t + 0.01);
    if (r > peak) { peak = r; tPeak = t; }
  }
  // Closure dip: quietest moment just before the onset.
  let pre = Infinity;
  for (let t = n.start - 0.07; t <= n.start - 0.015; t += 0.005) {
    pre = Math.min(pre, winRms(x, t, t + 0.01));
  }
  const dip = (Number.isFinite(pre) ? pre : 0) / peak;
  // Burst rise time: 10% -> 90% of the attack peak.
  let t10 = null;
  let t90 = null;
  for (let t = n.start - 0.02; t <= tPeak + 0.011; t += 0.0025) {
    const r = winRms(x, t, t + 0.01);
    if (t10 === null && r >= 0.1 * peak) t10 = t;
    if (r >= 0.9 * peak) { t90 = t; break; }
  }
  const rise = t10 !== null && t90 !== null ? Math.max(0, t90 - t10) : 0.1;
  // Vowel timbre: high-frequency ratio over the note body (spectral
  // centroid proxy — first-difference energy over signal energy).
  const b0 = Math.max(1, Math.round((n.start + 0.06) * ANALYSIS_SR));
  const b1 = Math.min(x.length, Math.round(Math.min(n.end, n.start + 0.25) * ANALYSIS_SR));
  let e = 1e-9;
  let d = 0;
  for (let i = b0; i < b1; i++) {
    e += x[i] * x[i];
    const dv = x[i] - x[i - 1];
    d += dv * dv;
  }
  const hf = Math.sqrt(d / e);
  return { dip, rise, hf };
}

/** Quietest-to-loudest energy ratio across a suspected note boundary. */
function boundaryDip(x, tBoundary) {
  let pre = Infinity;
  for (let t = tBoundary - 0.06; t <= tBoundary - 0.005; t += 0.005) {
    pre = Math.min(pre, winRms(x, t, t + 0.01));
  }
  let post = 1e-6;
  for (let t = tBoundary; t <= tBoundary + 0.06; t += 0.005) {
    post = Math.max(post, winRms(x, t, t + 0.01));
  }
  return (Number.isFinite(pre) ? pre : 0) / post;
}

// ---------------------------------------------------------------------------
// Minimum note length
// ---------------------------------------------------------------------------

/**
 * Tempo-aware minimum note length: notes shorter than the singer's declared
 * smallest note value can't be intentional. Absorb them into an adjacent
 * note when one is close in pitch; otherwise drop them.
 */
export function applyMinNote(notes, minDurSec) {
  const out = [];
  let absorbed = 0;
  let dropped = 0;
  for (let i = 0; i < notes.length; i++) {
    const n = notes[i];
    if (n.dur >= minDurSec) {
      out.push(n);
      continue;
    }
    const prev = out[out.length - 1];
    const next = notes[i + 1];
    const prevOk = prev && n.start - prev.end <= 0.08 && Math.abs(prev.midiFloat - n.midiFloat) <= 2;
    const nextOk = next && next.start - n.end <= 0.08 && Math.abs(next.midiFloat - n.midiFloat) <= 2;
    if (prevOk && (!nextOk || n.start - prev.end <= next.start - n.end)) {
      prev.end = Math.max(prev.end, n.end);
      prev.dur = prev.end - prev.start;
      absorbed++;
    } else if (nextOk) {
      next.start = n.start;
      next.dur = next.end - next.start;
      absorbed++;
    } else {
      dropped++;
    }
  }
  return { notes: out, absorbed, dropped };
}

// ---------------------------------------------------------------------------
// Auto-tune (melody simplification)
// ---------------------------------------------------------------------------

// Auto-tune thresholds at intensity 0.5 (the historic fixed behavior).
const BASE_ORNAMENT_DUR = 0.15; // notes shorter than this may be ornaments
const BASE_OUT_OF_KEY_DUR = 0.4; // longer out-of-key notes are deliberate
const BASE_MERGE_GAP = 0.05; // same-pitch fragments this close rejoin

/**
 * Melody auto-tune: reduces the transcription to its skeleton.
 * 1. Brief out-of-key notes snap to the nearest scale tone.
 * 2. Short notes within 2 semitones of a much longer neighbor are absorbed
 *    into it (grace notes, scoops, vibrato excursions).
 * 3. Adjacent same-pitch notes separated by tiny gaps merge.
 * Long out-of-key notes survive — blue notes and modulations are a choice,
 * not noise.
 *
 * `intensity` (0..1] scales how far each rule reaches: 0.5 reproduces the
 * historic fixed thresholds; 1 absorbs longer ornaments (up to ~0.3 s, from
 * shorter neighbors), corrects longer out-of-key notes (up to ~0.8 s), and
 * merges across wider gaps. Low values only touch the most obvious cases.
 */
export function simplifyNotes(notes, key, intensity = 0.5) {
  const k = intensity / 0.5;
  const ornamentMaxDur = Math.min(0.3, BASE_ORNAMENT_DUR * k);
  const outOfKeyMaxDur = Math.min(0.8, BASE_OUT_OF_KEY_DUR * k);
  const mergeGap = Math.min(0.12, BASE_MERGE_GAP * k);
  // How much longer a neighbor must be to absorb an ornament (2x at 0.5).
  const neighborRatio = Math.max(1.5, 3 - 2 * intensity);

  const degrees = key.mode === "major" ? [0, 2, 4, 5, 7, 9, 11] : [0, 2, 3, 5, 7, 8, 10, 11];
  const scale = new Set(degrees.map((d) => (key.tonic + d) % 12));

  let arr = notes.map((n) => ({ ...n }));

  // 1. Scale snapping for brief out-of-key notes.
  for (const n of arr) {
    const pc = pitchClass(n.midi);
    if (scale.has(pc) || n.dur >= outOfKeyMaxDur) continue;
    const upIn = scale.has((pc + 1) % 12);
    const downIn = scale.has((pc + 11) % 12);
    let shift;
    if (upIn && downIn) shift = n.midiFloat > n.midi ? 1 : -1; // toward how it was sung
    else shift = upIn ? 1 : -1;
    n.midi += shift;
    n.midiFloat = n.midi;
  }

  // 2. Ornament absorption (two passes so runs of short notes collapse).
  for (let pass = 0; pass < 2; pass++) {
    const res = [];
    for (let i = 0; i < arr.length; i++) {
      const n = arr[i];
      const prev = res[res.length - 1];
      const next = arr[i + 1];
      if (n.dur < ornamentMaxDur) {
        const prevOk = prev && prev.dur >= neighborRatio * n.dur &&
          Math.abs(prev.midi - n.midi) <= 2 && n.start - prev.end <= 0.08;
        const nextOk = next && next.dur >= neighborRatio * n.dur &&
          Math.abs(next.midi - n.midi) <= 2 && next.start - n.end <= 0.08;
        if (prevOk && (!nextOk || n.start - prev.end <= next.start - n.end)) {
          prev.end = Math.max(prev.end, n.end);
          prev.dur = prev.end - prev.start;
          continue;
        }
        if (nextOk) {
          next.start = n.start; // the main note starts where its ornament did
          next.dur = next.end - next.start;
          continue;
        }
      }
      res.push(n);
    }
    arr = res;
  }

  // 3. Merge identical adjacent pitches across tiny gaps.
  const merged = [];
  for (const n of arr) {
    const last = merged[merged.length - 1];
    if (last && last.midi === n.midi && n.start - last.end <= mergeGap) {
      last.end = Math.max(last.end, n.end);
      last.dur = last.end - last.start;
      last.amp = Math.max(last.amp ?? 0, n.amp ?? 0);
    } else {
      merged.push(n);
    }
  }
  return merged;
}
