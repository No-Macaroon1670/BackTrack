// Arrangement: anchors a 4/4 bar grid to the tracked beats (searching for
// the downbeat phase so pickups don't shift everything), fits a chord
// progression at half-bar granularity (Viterbi over diatonic chords with
// functional-harmony transition weights, metric-weighted melody evidence,
// and a chord-change penalty), then expands it into pad/bass/drum events
// per style. All event times come from the tracked beats, so the backing
// follows the singer's tempo drift — and all event velocities come from
// barDynamics, so it follows the singer's energy too: drums sit out the
// first bar, patterns thin out when the voice pulls back, phrase-ending
// rests get a drum fill into a crash, and the take closes on a held chord
// instead of stopping dead. Repeated phrases are found in form.js and
// harmonised identically, so the same material sounds like the same material.

import { detectForm, unifyRepeats } from "./form.js";

const NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
const FLAT_TONICS_MAJOR = new Set([5, 10, 3, 8, 1, 6]); // F Bb Eb Ab Db Gb
const FLAT_TONICS_MINOR = new Set([2, 7, 0, 5, 10, 3]); // d g c f bb eb

const MAJ = [0, 4, 7];
const MIN = [0, 3, 7];

// Chord palettes: offset = semitones above tonic.
const MAJOR_CHORDS = [
  { deg: "I", offset: 0, intervals: MAJ },
  { deg: "ii", offset: 2, intervals: MIN },
  { deg: "iii", offset: 4, intervals: MIN },
  { deg: "IV", offset: 5, intervals: MAJ },
  { deg: "V", offset: 7, intervals: MAJ },
  { deg: "vi", offset: 9, intervals: MIN },
];
const MINOR_CHORDS = [
  { deg: "i", offset: 0, intervals: MIN },
  { deg: "III", offset: 3, intervals: MAJ },
  { deg: "iv", offset: 5, intervals: MIN },
  { deg: "V", offset: 7, intervals: MAJ }, // harmonic-minor dominant
  { deg: "VI", offset: 8, intervals: MAJ },
  { deg: "VII", offset: 10, intervals: MAJ },
];

// Transition bonuses (functional harmony), keyed "from>to" by degree label.
const MAJOR_TRANSITIONS = {
  "V>I": 0.35, "IV>I": 0.2, "ii>V": 0.3, "IV>V": 0.25, "vi>IV": 0.25,
  "vi>ii": 0.2, "I>IV": 0.15, "I>V": 0.15, "I>vi": 0.15, "iii>vi": 0.2,
  "V>vi": 0.15,
};
const MINOR_TRANSITIONS = {
  "V>i": 0.35, "iv>V": 0.25, "VI>VII": 0.2, "VII>i": 0.25, "VI>iv": 0.2,
  "i>iv": 0.15, "i>VI": 0.15, "i>V": 0.15, "III>VI": 0.2, "iv>i": 0.15,
};
const SAME_CHORD_BONUS = 0.1;
const CHANGE_PENALTY = 0.18; // per chord change, so chords hold unless the melody insists
const MIDBAR_PENALTY = 0.1; // extra cost for changing inside a bar vs. on a barline

const STYLES = {
  pop: {
    // 16 steps per 4/4 measure (sixteenth-note grid).
    drums: { kick: [0, 8, 10], snare: [4, 12], hat: [0, 2, 4, 6, 8, 10, 12, 14] },
    bass: [
      { step: 0, len: 6, tone: "root" },
      { step: 6, len: 2, tone: "root" },
      { step: 8, len: 6, tone: "root" },
      { step: 14, len: 2, tone: "fifth" },
    ],
    pad: "sustain",
    swing: 0,
    seventh: false,
  },
  ballad: {
    drums: { kick: [0], snare: [8], hat: [0, 4, 8, 12] },
    bass: [
      { step: 0, len: 8, tone: "root" },
      { step: 8, len: 8, tone: "root" },
    ],
    pad: "arpeggio",
    swing: 0,
    seventh: false,
  },
  lofi: {
    drums: { kick: [0, 7, 10], snare: [4, 12], hat: [0, 2, 4, 6, 8, 10, 12, 14] },
    bass: [
      { step: 0, len: 8, tone: "root" },
      { step: 8, len: 8, tone: "fifth" },
    ],
    pad: "sustain",
    swing: 0.3, // delay applied to off-beat eighths
    seventh: true,
  },
  folk: {
    drums: { kick: [0, 8], hat: [0, 2, 4, 6, 8, 10, 12, 14], snare: [] },
    bass: [
      { step: 0, len: 8, tone: "root" },
      { step: 8, len: 8, tone: "fifth" },
    ],
    pad: "strum",
    swing: 0,
    seventh: false,
  },
};

// "D DU UDU" strum pattern: dir 1 = down (low string first), -1 = up, quieter.
const GUITAR_STRUM = [
  { step: 0, dir: 1, vel: 1.0, len: 4 },
  { step: 4, dir: 1, vel: 0.85, len: 2 },
  { step: 6, dir: -1, vel: 0.6, len: 4 },
  { step: 10, dir: -1, vel: 0.6, len: 2 },
  { step: 12, dir: 1, vel: 0.9, len: 2 },
  { step: 14, dir: -1, vel: 0.6, len: 2 },
];

/** Fixed-tempo fallback grid, used when the user overrides the detected tempo. */
export function uniformBeats(firstOnset, duration, bpm) {
  const interval = 60 / bpm;
  const beats = [];
  for (let t = firstOnset; t < duration + 5 * interval; t += interval) beats.push(t);
  return beats;
}

/**
 * @param {string|object} voicesOrInstrument either a chord-instrument string
 *   (legacy) or a per-role voices config: { chords: "keys"|"guitar"|"violin"|
 *   "voice", harmony: "follow"|<chord choice>|"off", bass: "acoustic"|
 *   "electric"|"synth", melody: "off"|<chord choice> }. The chords choice
 *   still determines the pad pattern (guitar strums, voices sustain).
 * @param {{harmony?: boolean}} opts legacy harmony toggle for the string form
 * @returns {{pads, bass, drums, harmony, melody, voices, chordSymbols, nMeasures, style, instrument, tuningCents, totalDur}}
 */
export function buildArrangement({ notes, key, beats, duration, tuningCents = 0 }, styleName, voicesOrInstrument = "keys", { harmony = true, flourish = 0, songForm = true } = {}) {
  const voices = typeof voicesOrInstrument === "string"
    ? { chords: voicesOrInstrument, harmony: harmony ? "follow" : "off", bass: "acoustic", melody: "off" }
    : { chords: "keys", harmony: "follow", bass: "acoustic", melody: "off", ...voicesOrInstrument };
  const instrument = voices.chords;
  const style = STYLES[styleName];
  const phase = findDownbeatPhase(notes, beats);
  const bars = buildBars(beats, phase, notes, duration);

  const palette = key.mode === "major" ? MAJOR_CHORDS : MINOR_CHORDS;
  const transitions = key.mode === "major" ? MAJOR_TRANSITIONS : MINOR_TRANSITIONS;

  const weights = segmentWeights(notes, bars); // two half-bar segments per bar
  const path = viterbiChords(weights, palette, transitions, key.tonic);
  const dyn = barDynamics(notes, bars);

  const segChords = path.map((idx) => {
    const c = palette[idx];
    const intervals = style.seventh
      ? [...c.intervals, c.intervals === MIN ? 10 : 11]
      : c.intervals;
    return {
      rootPc: (key.tonic + c.offset) % 12,
      intervals,
      quality: c.intervals === MIN ? "min" : "maj",
      deg: c.deg,
    };
  });

  // Song form: make the same phrase sound like the same phrase. Repeats are
  // harmonised identically, and later occurrences are played a little fuller
  // so the take builds instead of restating.
  const form = songForm ? detectForm(notes, bars) : { phrases: [], nLabels: 0 };
  if (songForm && form.phrases.length) {
    const unified = unifyRepeats(form, segChords);
    if (unified) console.debug(`Song form: ${form.nLabels} distinct phrases, ${unified} segments unified across repeats`);
    applyFormArc(form, bars, dyn);
  }

  const anchors = chordAnchors(notes, bars, segChords);
  const colors = segmentColors(weights, segChords);
  const events = expandEvents(segChords, bars, style, instrument, dyn, anchors, colors);
  events.harmony = voices.harmony !== "off" ? genHarmonyLine(notes, segChords, bars, key, dyn) : [];
  events.melody = voices.melody !== "off" ? genMelodyDouble(notes, bars, dyn) : [];
  if (events.melody.length && flourish > 0) {
    events.melody = ornamentMelody(events.melody, bars, key, dyn, Math.min(1, flourish));
  }
  if (voices.bass === "off") events.bass = [];
  const chordSymbols = bars.map((bar, b) => {
    const a = segChords[2 * b];
    const c2 = segChords[2 * b + 1];
    const nameA = chordName(a, key, style.seventh) + colorSuffix(colors[2 * b]);
    return {
      measure: b,
      name: sameChord(a, c2)
        ? nameA
        : `${nameA} · ${chordName(c2, key, style.seventh)}${colorSuffix(colors[2 * b + 1])}`,
    };
  });

  return {
    ...events,
    voices,
    chordSymbols,
    bars, // beat boundaries per bar, for MIDI export's tempo map
    nMeasures: bars.length,
    style: styleName,
    instrument,
    tuningCents,
    // The ending chord rings past the last bar (see expandEvents).
    totalDur: bars[bars.length - 1].bt[4] + 6 * barBeat(bars[bars.length - 1]) + 1.2,
  };
}

function sameChord(a, b) {
  return a.rootPc === b.rootPc && a.quality === b.quality;
}

// ---------------------------------------------------------------------------
// Chord color: tension tones earned by the melody
// ---------------------------------------------------------------------------

/**
 * Picks at most one color tone per half-bar segment, and only with evidence:
 * the singer must actually have leaned on that tone over this chord (the
 * normalized pitch-class weights the Viterbi already uses). The exception is
 * the cadential dominant seventh — V resolving to I gets its b7 for free,
 * because that pull IS the cadence. Chord CHOICE is never affected; this
 * only enriches how the chosen chord is voiced.
 */
function segmentColors(weights, segChords) {
  return segChords.map((c, s) => {
    const w = weights[s];
    if (!w) return null;
    const pc = (iv) => (c.rootPc + iv) % 12;
    const next = segChords[s + 1];
    const isTriad = c.intervals.length === 3;
    if (c.deg === "V" && next && next.deg === "I" && isTriad) {
      return { type: "add", iv: 10 }; // cadential V7
    }
    if (!isTriad) return null; // lofi's 7th chords are colorful enough
    if (c.quality === "maj" && w[pc(5)] >= 0.22 && w[pc(4)] < 0.05) {
      return { type: "sus4" }; // the singer avoids the 3rd and leans on the 4th
    }
    if (w[pc(2)] >= 0.18) return { type: "add", iv: 14 }; // add9
    if ((c.deg === "I" || c.deg === "IV") && w[pc(11)] >= 0.18) {
      return { type: "add", iv: 11 }; // maj7
    }
    return null;
  });
}

function colorSuffix(color) {
  if (!color) return "";
  if (color.type === "sus4") return "sus4";
  return color.iv === 10 ? "7" : color.iv === 11 ? "maj7" : "add9";
}

/** Applies a color to a voicing: sus4 replaces the 3rd, adds sit on top. */
function applyColor(voicing, chord, color) {
  if (!color) return voicing;
  const v = [...voicing];
  if (color.type === "sus4") {
    const thirdPc = (chord.rootPc + 4) % 12; // sus4 is only chosen for major
    const i = v.findIndex((m) => ((m % 12) + 12) % 12 === thirdPc);
    if (i >= 0) v[i] += 1; // major third -> fourth
    return v.sort((a, b) => a - b);
  }
  const top = v[v.length - 1];
  const colorPc = (chord.rootPc + color.iv) % 12;
  let note = top + ((((colorPc - top) % 12) + 12) % 12);
  if (note === top) note += 12;
  if (note > 79) note -= 12;
  if (!v.includes(note)) v.push(note);
  return v.sort((a, b) => a - b);
}

/**
 * Singers often start on a pickup, so the first note isn't necessarily
 * beat 1. Try each of the four beat phases and keep the one where note
 * onsets land most heavily on beats 1 (weight 2) and 3 (weight 1).
 */
function findDownbeatPhase(notes, beats) {
  const scores = [0, 0, 0, 0];
  for (const n of notes) {
    // Nearest beat to this onset.
    let lo = 0, hi = beats.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (beats[mid] < n.start) lo = mid;
      else hi = mid;
    }
    const i = Math.abs(beats[lo] - n.start) <= Math.abs(beats[hi] - n.start) ? lo : hi;
    if (Math.abs(beats[i] - n.start) > 0.12) continue;
    const w = Math.min(n.dur, 1);
    for (let p = 0; p < 4; p++) {
      const pos = (((i - p) % 4) + 4) % 4;
      scores[p] += w * (pos === 0 ? 2 : pos === 2 ? 1 : 0);
    }
  }
  let best = 0;
  for (let p = 1; p < 4; p++) if (scores[p] > scores[best]) best = p;
  return best;
}

/** Groups beats into 4/4 bars; each bar holds its five beat boundaries. */
function buildBars(beats, phase, notes, duration) {
  const bars = [];
  for (let i = phase; i + 4 < beats.length; i += 4) {
    if (beats[i] >= duration) break;
    bars.push({ bt: beats.slice(i, i + 5) });
  }
  // Don't open with bars of silence before the first note.
  const firstNote = notes[0].start;
  while (bars.length > 1 && bars[0].bt[4] <= firstNote + 0.05) bars.shift();
  if (bars.length === 0) bars.push({ bt: beats.slice(phase, phase + 5) });
  return bars;
}

function barBeat(bar) {
  return (bar.bt[4] - bar.bt[0]) / 4;
}

/**
 * Shapes the take across repeats: the first time a phrase is heard it is
 * played back a little, and each return is played fuller. That is the
 * cheapest honest version of an arrangement arc — the band commits more as
 * the song establishes itself — and it rides on the dynamics layer that
 * already exists, so drums thin and thicken accordingly.
 */
function applyFormArc(form, bars, dyn) {
  // Only shape material that actually returns. A take with no repeats (a
  // single pass of a tune, which is what most short hums are) must come out
  // byte-identical to form-off — otherwise the feature quietly re-levels
  // every arrangement it can't find structure in.
  const counts = new Map();
  for (const p of form.phrases) counts.set(p.label, (counts.get(p.label) ?? 0) + 1);
  if (![...counts.values()].some((n) => n > 1)) return;
  for (const p of form.phrases) {
    if ((counts.get(p.label) ?? 0) < 2) continue;
    const gain = p.occurrence === 0 ? 0.92 : Math.min(1.12, 1 + 0.06 * p.occurrence);
    for (let b = p.startBar; b <= p.endBar && b < dyn.intensity.length; b++) {
      dyn.intensity[b] = Math.max(0.3, Math.min(1, dyn.intensity[b] * gain));
    }
  }
}

/**
 * Where a chord CHANGES, singers rarely land exactly on the barline — the
 * phrase arrives a breath early or late. A real accompanist moves the chord
 * with the singer while the rhythm section holds the grid. For each
 * half-bar boundary whose chord differs from the previous segment's (plus
 * the very first chord), this returns the nearest note onset within ±120 ms
 * of the grid time — or the grid time itself. Only the sustained chord
 * voice consumes these; bass, drums, strums and arpeggios stay on the grid.
 */
const SNAP_WINDOW = 0.12;

function chordAnchors(notes, bars, segChords) {
  const bounds = [];
  for (const bar of bars) { bounds.push(bar.bt[0]); bounds.push(bar.bt[2]); }
  bounds.push(bars[bars.length - 1].bt[4]);

  return bounds.map((t, s) => {
    const isLast = s === bounds.length - 1;
    const changes = s === 0 || (!isLast && !sameChord(segChords[s - 1], segChords[s]));
    if (!changes || isLast) return t;
    let best = null;
    for (const n of notes) {
      const d = Math.abs(n.start - t);
      if (d <= SNAP_WINDOW && (best === null || d < Math.abs(best - t))) best = n.start;
    }
    // Shifts under ~15 ms aren't worth deviating from the grid for.
    return best !== null && Math.abs(best - t) > 0.015 ? best : t;
  });
}

// ---------------------------------------------------------------------------
// Harmony voice
// ---------------------------------------------------------------------------

const SCALE_IV = { major: [0, 2, 4, 5, 7, 9, 11], minor: [0, 2, 3, 5, 7, 8, 10] };
const THIRD_BELOW = [3, 4, 5];
const SIXTH_BELOW = [8, 9, 10];
const THIRD_ABOVE = [-3, -4, -5];
const HARMONY_FLOOR = 40; // E2: thirds below a low voice get muddy fast

/** Chord tone at one of the offsets below `midi`, else a scale tone there. */
function toneNear(midi, offsets, chordPcs, scalePcs) {
  for (const off of offsets) {
    if (chordPcs.has((((midi - off) % 12) + 12) % 12)) return midi - off;
  }
  for (const off of offsets) {
    if (scalePcs.has((((midi - off) % 12) + 12) % 12)) return midi - off;
  }
  return null;
}

/**
 * The melody-shadow counter-voice: each sustained sung note gets a companion
 * a third below (occasionally a sixth), chord tones first, scale tones as
 * the fallback — and when "below" would sink under the harmony floor (low
 * voices), a third above instead. It keeps the singer's own timing, so it
 * inherits all the phrasing and can never clash rhythmically; fast runs are
 * left alone. Velocities follow the band's per-bar dynamics.
 */
function genHarmonyLine(notes, segChords, bars, key, dyn) {
  const scalePcs = new Set(SCALE_IV[key.mode].map((i) => (key.tonic + i) % 12));
  const rng = mulberry32(0x4a72 ^ (notes.length * 977));
  const minDur = Math.max(0.2, 0.5 * barBeat(bars[0]));
  const out = [];
  for (const n of notes) {
    if (n.end - n.start < minDur) continue;
    const b = bars.findIndex((bar) => n.start >= bar.bt[0] && n.start < bar.bt[4]);
    if (b < 0) continue;
    const seg = segChords[2 * b + (n.start >= bars[b].bt[2] ? 1 : 0)];
    const chordPcs = new Set(seg.intervals.map((iv) => (seg.rootPc + iv) % 12));
    const midi = Math.round(n.midiFloat ?? n.midi);

    let p = null;
    if (rng() < 0.25) p = toneNear(midi, SIXTH_BELOW, chordPcs, scalePcs);
    if (p === null) p = toneNear(midi, THIRD_BELOW, chordPcs, scalePcs);
    if (p !== null && p < HARMONY_FLOOR) p = null;
    if (p === null) p = toneNear(midi, THIRD_ABOVE, chordPcs, scalePcs);
    if (p === null) continue;

    // The sung timing IS the humanization — only velocity varies.
    const iVel = 0.55 + 0.45 * (dyn.intensity[b] ?? 0.75);
    out.push({ t: n.start, dur: (n.end - n.start) * 0.95, midi: p, vel: iVel * (0.95 + rng() * 0.1) });
  }
  return out;
}

/**
 * The melody-double role: an instrument playing the sung line itself, at
 * the singer's exact timing and the band's per-bar dynamics. (The flourish
 * control from the roadmap will ornament this stream.)
 */
function genMelodyDouble(notes, bars, dyn) {
  return notes.flatMap((n) => {
    const b = bars.findIndex((bar) => n.start >= bar.bt[0] && n.start < bar.bt[4]);
    if (b < 0) return [];
    return [{
      t: n.start,
      dur: (n.end - n.start) * 0.98,
      midi: Math.round(n.midiFloat ?? n.midi),
      vel: 0.55 + 0.45 * (dyn.intensity[b] ?? 0.75),
    }];
  });
}

/**
 * Flourish: ornaments the melody-double stream — never the vocal or the
 * transcribed melody. Two figures, both scale-constrained and skipped in
 * bars where the singer is already busy (the arp-clash lesson):
 *   - passing tones: a leap of a third or more toward the next note gets
 *     diatonic steps in its tail (two of them on big leaps at high amounts)
 *   - neighbor figures: a held note dips to an adjacent scale tone and
 *     returns just before it ends
 * `amount` 0..1 scales both how often figures fire and how elaborate they
 * get. Seeded PRNG: the same take + slider renders identically.
 */
function ornamentMelody(events, bars, key, dyn, amount) {
  const scalePcs = SCALE_IV[key.mode].map((i) => (key.tonic + i) % 12);
  const inScale = (m) => scalePcs.includes(((m % 12) + 12) % 12);
  const rng = mulberry32(0xf10c ^ (events.length * 271));
  const beat = barBeat(bars[0]);
  const out = [];

  for (let i = 0; i < events.length; i++) {
    const cur = { ...events[i] };
    const next = events[i + 1];
    const b = bars.findIndex((bar) => cur.t >= bar.bt[0] && cur.t < bar.bt[4]);
    const busy = b >= 0 && dyn.busy[b] > 4;

    // Passing tones: fill a leap's tail with diatonic steps toward the next note.
    const leap = next ? next.midi - cur.midi : 0;
    if (!busy && next && Math.abs(leap) >= 3 && cur.dur >= 0.7 * beat && rng() < amount * 0.9) {
      const between = [];
      for (let m = cur.midi + Math.sign(leap); m !== next.midi; m += Math.sign(leap)) {
        if (inScale(m)) between.push(m);
      }
      if (between.length) {
        const nPass = between.length >= 2 && Math.abs(leap) >= 7 && amount > 0.6 && rng() < 0.6 ? 2 : 1;
        const picks = nPass === 1
          ? [between[Math.floor(between.length / 2)]]
          : [between[Math.floor(between.length / 3)], between[Math.floor((2 * between.length) / 3)]];
        const d = Math.min(0.25 * beat, cur.dur * 0.22);
        cur.dur -= d * nPass;
        out.push(cur);
        picks.forEach((midi, k) => {
          out.push({ t: cur.t + cur.dur + k * d, dur: d * 0.92, midi, vel: cur.vel * 0.75 });
        });
        continue;
      }
    }

    // Neighbor figure: a held note dips to an adjacent scale tone and returns.
    if (!busy && cur.dur >= 1.5 * beat && rng() < amount * 0.7) {
      let nb = cur.midi + (rng() < 0.65 ? 1 : -1);
      while (!inScale(nb)) nb += Math.sign(nb - cur.midi);
      const d = Math.min(0.25 * beat, cur.dur * 0.15);
      out.push({ ...cur, dur: cur.dur - 2 * d });
      out.push({ t: cur.t + cur.dur - 2 * d, dur: d * 0.92, midi: nb, vel: cur.vel * 0.7 });
      out.push({ t: cur.t + cur.dur - d, dur: d * 0.92, midi: cur.midi, vel: cur.vel * 0.8 });
      continue;
    }

    out.push(cur);
  }
  return out;
}

/** Deterministic PRNG so humanization renders identically across runs. */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The "breathing" layer: how the arrangement responds to the performance.
 *
 * - intensity[b] 0.35..1 per bar: the vocal's own energy (amplitude-weighted
 *   coverage), smoothed over neighbours and normalized across the take, so
 *   the band leans in when the singer does and thins out when they pull back.
 *   A flat performance maps to a steady 0.75 rather than noise.
 * - fills: bars that end a sung phrase (a rest of at least ~1.3 beats before
 *   the next entry) get a drum fill leading into the next phrase's downbeat,
 *   which lands with a crash. The final bar always fills into the ending.
 * - drumsFrom: drums sit out the first bar (pads + bass introduce the song)
 *   on takes long enough to afford it.
 */
function barDynamics(notes, bars) {
  const nB = bars.length;
  const raw = bars.map((bar) => {
    let e = 0;
    for (const n of notes) {
      const span = Math.min(n.end, bar.bt[4]) - Math.max(n.start, bar.bt[0]);
      if (span > 0) e += span * (n.amp ?? 0.8);
    }
    return e / (bar.bt[4] - bar.bt[0]);
  });
  const smooth = raw.map((_, b) => {
    let sum = 0, k = 0;
    for (const v of [raw[b - 1], raw[b], raw[b + 1]]) {
      if (v !== undefined) { sum += v; k++; }
    }
    return sum / k;
  });
  const lo = Math.min(...smooth), hi = Math.max(...smooth);
  const intensity = smooth.map((v) =>
    hi - lo < 0.1 ? 0.75 : 0.35 + 0.65 * ((v - lo) / (hi - lo)));

  const fills = new Set();
  const beat = barBeat(bars[0]);
  for (let i = 0; i + 1 < notes.length; i++) {
    if (notes[i + 1].start - notes[i].end < 1.3 * beat) continue;
    const b = bars.findIndex((bar) => notes[i + 1].start >= bar.bt[0] && notes[i + 1].start < bar.bt[4]);
    if (b > 1) fills.add(b - 1); // fill the bar leading into the new phrase
  }
  fills.add(nB - 1); // and always lead into the ending
  // Onset density per bar: rhythmic figures (arpeggios) yield to a busy voice.
  const busy = bars.map((bar) => notes.filter((n) => n.start >= bar.bt[0] && n.start < bar.bt[4]).length);
  return { intensity, fills, drumsFrom: nB >= 4 ? 1 : 0, busy };
}

/**
 * Melody evidence per half-bar segment: pitch-class weights from note
 * overlap, scaled up for notes that land on strong beats (the harmony
 * carriers) and scaled down for likely passing tones.
 */
function segmentWeights(notes, bars) {
  const segs = [];
  for (const bar of bars) {
    segs.push({ start: bar.bt[0], mid: bar.bt[1], end: bar.bt[2] });
    segs.push({ start: bar.bt[2], mid: bar.bt[3], end: bar.bt[4] });
  }
  const weights = segs.map(() => new Array(12).fill(0));

  notes.forEach((n, ni) => {
    const pc = ((n.midi % 12) + 12) % 12;

    // Passing-tone discount: a short note approached and left stepwise in
    // the same direction is ornamental, not harmonic.
    const prev = notes[ni - 1], next = notes[ni + 1];
    let factor = 1;
    if (prev && next && n.dur < 0.28 &&
        Math.abs(prev.midi - n.midi) <= 2 && Math.abs(next.midi - n.midi) <= 2 &&
        n.midi !== prev.midi && n.midi !== next.midi &&
        Math.sign(n.midi - prev.midi) === Math.sign(next.midi - n.midi)) {
      factor = 0.4;
    }

    for (let si = 0; si < segs.length; si++) {
      const seg = segs[si];
      const span = Math.min(n.end, seg.end) - Math.max(n.start, seg.start);
      if (span <= 0) continue;
      let w = span * factor;
      const eighth = (seg.mid - seg.start) / 2;
      // A note pushed up to an eighth AHEAD of the boundary and held across
      // it is an anticipation: it belongs to this segment's chord and gets
      // the downbeat bonus here...
      const anticipatesThis = n.start >= seg.start - eighth && n.start < seg.start && n.end > seg.start + 0.05;
      if (Math.abs(n.start - seg.start) < 0.07 || anticipatesThis) w *= 2; // beat 1/3
      else if (Math.abs(n.start - seg.mid) < 0.07) w *= 1.4; // beat 2/4
      // ...and mostly does NOT vote for the segment it merely started in.
      if (n.start >= seg.start && n.start >= seg.end - eighth && n.end > seg.end + 0.05) w *= 0.4;
      weights[si][pc] += w;
    }
  });

  // Normalize each segment to sum 1 so emission and transition scores stay comparable.
  for (const w of weights) {
    const sum = w.reduce((a, b) => a + b, 0);
    if (sum > 0) for (let i = 0; i < 12; i++) w[i] /= sum;
  }
  return weights;
}

function emissionScore(w, chord, tonic) {
  const sum = w.reduce((a, b) => a + b, 0);
  if (sum === 0) return 0; // empty segment: let transitions decide
  const tones = new Set(chord.intervals.map((iv) => (tonic + chord.offset + iv) % 12));
  const rootPc = (tonic + chord.offset) % 12;
  let inTone = 0, outTone = 0;
  for (let pc = 0; pc < 12; pc++) {
    if (tones.has(pc)) inTone += w[pc];
    else outTone += w[pc];
  }
  return inTone - 0.7 * outTone + 0.3 * w[rootPc];
}

function viterbiChords(weights, palette, transitions, tonic) {
  const nS = weights.length;
  const nC = palette.length;
  const dp = Array.from({ length: nS }, () => new Array(nC).fill(-Infinity));
  const back = Array.from({ length: nS }, () => new Array(nC).fill(0));

  for (let c = 0; c < nC; c++) {
    dp[0][c] = emissionScore(weights[0], palette[c], tonic) + (c === 0 ? 0.3 : 0);
  }
  for (let s = 1; s < nS; s++) {
    const midBar = s % 2 === 1; // odd segment = second half of a bar
    for (let c = 0; c < nC; c++) {
      const em = emissionScore(weights[s], palette[c], tonic);
      for (let p = 0; p < nC; p++) {
        let tr;
        if (p === c) {
          tr = SAME_CHORD_BONUS;
        } else {
          const bonus = transitions[`${palette[p].deg}>${palette[c].deg}`] ?? 0;
          tr = bonus - CHANGE_PENALTY - (midBar ? MIDBAR_PENALTY : 0);
        }
        const score = dp[s - 1][p] + tr + em + (s === nS - 1 && c === 0 ? 0.5 : 0);
        if (score > dp[s][c]) { dp[s][c] = score; back[s][c] = p; }
      }
    }
  }

  let best = 0;
  for (let c = 1; c < nC; c++) if (dp[nS - 1][c] > dp[nS - 1][best]) best = c;
  const path = new Array(nS);
  path[nS - 1] = best;
  for (let s = nS - 1; s > 0; s--) path[s - 1] = back[s][path[s]];
  return path;
}

/** Time of 16th-note step `step` (0..16+, may be fractional) within a bar. */
function stepTime(bar, step) {
  const bt = bar.bt;
  if (step >= 16) return bt[4] + ((step - 16) / 4) * (bt[4] - bt[3]);
  const b = Math.floor(step / 4);
  const frac = (step % 4) / 4;
  return bt[b] + frac * (bt[b + 1] - bt[b]);
}

function expandEvents(segChords, bars, style, instrument, dyn, anchors, colors) {
  const pads = [];
  const bass = [];
  const drums = [];
  let prevVoicing = null;
  // Bar-level color choices (bass octave alternation) from a seeded PRNG,
  // consumed once per bar so builds stay deterministic.
  const rngBar = mulberry32(0xba55 ^ (bars.length * 419));

  bars.forEach((bar, b) => {
    const cA = segChords[2 * b];
    const cB = segChords[2 * b + 1];
    const split = !sameChord(cA, cB);
    const iVel = 0.55 + 0.45 * dyn.intensity[b]; // per-bar dynamic level
    const octaveAlt = rngBar() < 0.3; // this bar's "fifth" becomes the octave
    const padsBase = pads.length;

    const vA = applyColor(voiceChord(cA.rootPc, cA.intervals, prevVoicing), cA, colors[2 * b]);
    const vB = split ? applyColor(voiceChord(cB.rootPc, cB.intervals, vA), cB, colors[2 * b + 1]) : vA;
    prevVoicing = vB;
    const voicingAt = (step) => (step < 8 ? vA : vB);
    const chordAt = (step) => (step < 8 ? cA : cB);

    // --- Pads / keys / guitar / strings ---
    // Guitar can't sustain (strums/fingerpicks); bowed strings can't strum
    // or comp (always sustained legato).
    const padMode = instrument === "guitar"
      ? (style.pad === "arpeggio" ? "fingerpick" : "guitarStrum")
      : instrument === "violin" || instrument === "voice" ? "sustain"
      : style.pad;
    if (padMode === "sustain") {
      // Chord-change boundaries follow the singer (see chordAnchors).
      const t0 = anchors[2 * b];
      const tMid = anchors[2 * b + 1];
      const t1 = anchors[2 * b + 2];
      if (split) {
        pads.push({ t: t0, dur: tMid - t0, midis: vA, strum: 0, vel: 1 });
        pads.push({ t: tMid, dur: t1 - tMid, midis: vB, strum: 0, vel: 1 });
      } else {
        pads.push({ t: t0, dur: t1 - t0, midis: vA, strum: 0, vel: 1 });
      }
    } else if (padMode === "strum") {
      for (const step of [0, 4, 8, 12]) {
        const t = stepTime(bar, step);
        pads.push({ t, dur: stepTime(bar, step + 3.5) - t, midis: voicingAt(step), strum: 0.015, vel: 1 });
      }
    } else if (padMode === "guitarStrum") {
      for (const s of GUITAR_STRUM) {
        const v = voicingAt(s.step);
        const strumVoicing = [...v, v[0] + 12]; // doubled root on top for sparkle
        const t = stepTime(bar, s.step);
        pads.push({
          t,
          dur: stepTime(bar, s.step + s.len) - t,
          midis: s.dir > 0 ? strumVoicing : [...strumVoicing].reverse(),
          strum: 0.014,
          vel: s.vel,
        });
      }
    } else if (padMode === "arpeggio" || padMode === "fingerpick") {
      // A motoric eighth-note figure fights a busy melody (the MIDI Enhancer
      // lesson: fixed-rate arps miss on style clash). When the voice sings
      // more than about one onset per beat this bar, hold the chord instead
      // and let the melody carry the motion.
      if (dyn.busy[b] > 4) {
        if (split) {
          pads.push({ t: bar.bt[0], dur: bar.bt[2] - bar.bt[0], midis: vA, strum: padMode === "fingerpick" ? 0.015 : 0, vel: 1 });
          pads.push({ t: bar.bt[2], dur: bar.bt[4] - bar.bt[2], midis: vB, strum: padMode === "fingerpick" ? 0.015 : 0, vel: 1 });
        } else {
          pads.push({ t: bar.bt[0], dur: bar.bt[4] - bar.bt[0], midis: vA, strum: padMode === "fingerpick" ? 0.015 : 0, vel: 1 });
        }
      } else {
        const seq = [0, 1, 2, 3, 2, 1, 2, 3]; // indices into voicing (wrapping)
        for (let s = 0; s < 8; s++) {
          const step = s * 2;
          const v = voicingAt(step);
          const note = v[seq[s] % v.length] + (seq[s] >= v.length ? 12 : 0);
          const t = stepTime(bar, step);
          pads.push({ t, dur: stepTime(bar, step + 3) - t, midis: [note], strum: 0, vel: s % 2 ? 0.75 : 1 });
        }
      }
    }

    // Dynamics: the whole bar's pads follow the vocal's energy.
    for (let i = padsBase; i < pads.length; i++) pads[i].vel *= iVel;

    // --- Bass ---
    for (const p of style.bass) {
      const chord = chordAt(p.step);
      const rootMidi = 36 + chord.rootPc;
      const fifthMidi = rootMidi + 7 <= 47 ? rootMidi + 7 : rootMidi - 5;
      const t = stepTime(bar, p.step);
      bass.push({
        t,
        dur: (stepTime(bar, p.step + p.len) - t) * 0.95,
        midi: p.tone === "fifth" ? (octaveAlt ? rootMidi + 12 : fifthMidi) : rootMidi,
        vel: iVel,
      });
    }

    // --- Drums ---
    if (b >= dyn.drumsFrom) {
      const quiet = dyn.intensity[b] < 0.45;
      const fill = dyn.fills.has(b);
      for (const [type, steps] of Object.entries(style.drums)) {
        for (const step of steps) {
          if (fill && step >= 10 && type !== "kick") continue; // clear room for the fill
          if (quiet && type === "hat" && step % 4 !== 0) continue; // pull back: quarter hats
          if (quiet && type === "kick" && step !== 0 && step !== 8) continue; // and a plainer kick
          let t = stepTime(bar, step);
          // Swing: delay off-beat eighths (steps 2, 6, 10, 14).
          if (style.swing > 0 && step % 4 === 2) {
            t += style.swing * (stepTime(bar, step + 1) - t);
          }
          // Metric accent: off-beat hats sit a touch under the on-beat ones.
          drums.push({ t, type, vel: type === "hat" && step % 4 !== 0 ? iVel * 0.85 : iVel });
        }
      }
      if (fill) {
        // Snare fill rising into the next downbeat.
        const steps = dyn.intensity[b] > 0.6 ? [10, 12, 13, 14, 15] : [12, 14, 15];
        steps.forEach((s, k) => {
          drums.push({ t: stepTime(bar, s), type: "snare", vel: iVel * (0.45 + 0.55 * (k + 1) / steps.length) });
        });
      }
      if (b > 0 && dyn.fills.has(b - 1)) {
        drums.push({ t: bar.bt[0], type: "crash", vel: Math.min(1, iVel + 0.1) });
      }
    }
  });

  // Humanization: a real band isn't sample-accurate. A few milliseconds of
  // looseness and a little velocity variation on the comping instruments;
  // drums stay tightest — they are the grid. Seeded PRNG, so the same take
  // renders identically every time (verification depends on this). The
  // ending events below are pushed after this pass and stay exact.
  const rng = mulberry32(0x5eed ^ (bars.length * 7349) ^ (segChords.length * 131));
  const j = (amt) => (rng() * 2 - 1) * amt;
  for (const p of pads) { p.t = Math.max(0, p.t + j(0.008)); p.vel *= 1 + j(0.06); }
  for (const n of bass) { n.t = Math.max(0, n.t + j(0.005)); n.vel *= 1 + j(0.05); }
  for (const d of drums) { d.t = Math.max(0, d.t + j(d.type === "hat" ? 0.004 : 0.003)); }

  // Ending: the last chord rings out on the downbeat after the final bar,
  // under a kick + crash, instead of the track just stopping.
  const endBar = bars[bars.length - 1];
  const endT = endBar.bt[4];
  const beat = barBeat(endBar);
  const last = segChords[segChords.length - 1];
  const vEnd = voiceChord(last.rootPc, last.intervals, prevVoicing);
  pads.push({ t: endT, dur: 6 * beat, midis: vEnd, strum: instrument === "guitar" ? 0.018 : 0, vel: 0.95 });
  bass.push({ t: endT, dur: 5 * beat, midi: 36 + last.rootPc, vel: 0.95 });
  drums.push({ t: endT, type: "kick", vel: 0.9 });
  drums.push({ t: endT, type: "crash", vel: 0.85 });

  return { pads, bass, drums };
}

/**
 * Voices a chord into the G3-C5 register, choosing the inversion whose
 * centroid is closest to the previous chord's (smooth voice leading).
 */
function voiceChord(rootPc, intervals, prevVoicing) {
  let root = 48 + rootPc; // C3..B3
  if (root > 55) root -= 12;
  const base = intervals.map((iv) => root + iv);

  const candidates = [];
  for (let inv = 0; inv < base.length; inv++) {
    const notes = base.map((n, i) => (i < inv ? n + 12 : n)).sort((a, b) => a - b);
    candidates.push(notes);
  }
  const prevMean = prevVoicing
    ? prevVoicing.reduce((a, b) => a + b, 0) / prevVoicing.length
    : 62;
  let best = candidates[0];
  let bestCost = Infinity;
  for (const cand of candidates) {
    const mean = cand.reduce((a, b) => a + b, 0) / cand.length;
    const cost = Math.abs(mean - 62) + 0.8 * Math.abs(mean - prevMean);
    if (cost < bestCost) { bestCost = cost; best = cand; }
  }
  return best;
}

function chordName(chord, key, seventh) {
  const useFlats = key.mode === "major"
    ? FLAT_TONICS_MAJOR.has(key.tonic)
    : FLAT_TONICS_MINOR.has(key.tonic);
  const names = useFlats ? NOTE_NAMES_FLAT : NOTE_NAMES_SHARP;
  let name = names[chord.rootPc];
  if (chord.quality === "min") name += "m";
  if (seventh) name += chord.quality === "min" ? "7" : "maj7";
  return name;
}

export function keyName(key) {
  const useFlats = key.mode === "major"
    ? FLAT_TONICS_MAJOR.has(key.tonic)
    : FLAT_TONICS_MINOR.has(key.tonic);
  const names = useFlats ? NOTE_NAMES_FLAT : NOTE_NAMES_SHARP;
  return `${names[key.tonic]} ${key.mode}`;
}
