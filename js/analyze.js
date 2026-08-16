// Vocal analysis orchestration. Runs the pipeline:
//   transcribe (Basic Pitch neural, YIN fallback)
//     -> YIN cross-validation -> onset snap -> syllable lock
//     -> tempo pre-estimate -> min-note filter
//     -> tuning offset -> key -> auto-tune
//     -> final beat grid -> note-anchored grid.
// The individual stages live in focused modules (dsp, notes, rhythm, key);
// this file wires them together and hosts the YIN fallback engine used when
// the neural model can't load.

import { transcribeBasicPitch, getBackend, BASIC_PITCH_SR, recoverGapNotes } from "./transcribe.js";
import {
  ANALYSIS_SR, FRAME_SIZE, HOP, toMono, resample, median, createYin, yinFrame,
} from "./dsp.js";
import {
  refineNote, octaveCorrect, consolidateRepeats, snapNoteOnsets, snapNoteOffsets,
  applySyllableLock, applyMinNote, simplifyNotes,
} from "./notes.js";
import { detectBeats, noteAnchoredGrid, foldTempo, bestGridTempo } from "./rhythm.js";
import { detectKey, estimateTuning } from "./key.js";

// detectKey is used directly by app.js (MIDI-upload path); re-export so its
// import site doesn't need to know which module it lives in.
export { detectKey };

// YIN fallback-engine segmentation parameters.
const MIN_NOTE_FRAMES = 5; // ~80 ms
const NOTE_SPLIT_SEMITONES = 0.7;
const MAX_UNVOICED_GAP = 3; // frames of dropout tolerated inside one note

const tick = () => new Promise((r) => setTimeout(r, 0));

/**
 * @param {AudioBuffer} audioBuffer the recorded vocal
 * @param {(p: number) => void} onProgress 0..1
 * @param {object} options transcription/cleaning toggles
 * @param {object|null} modelCache per-take Basic Pitch output (see transcribe.js)
 * @returns {Promise<{notes, key, tempo, beats, tuningCents, engine, backend, lockInfo, modelCache, firstOnset, duration}>}
 */
export async function analyzeVocal(
  audioBuffer,
  onProgress = () => {},
  { sensitivity = 0.5, snapOnsets = true, followNotes = true, autoTune = 0, syllableLock = false, minNote = 0.25, octaveFix = false, adaptivePitch = true, consolidate = true, gapRecovery = true, snapOffsets = false, pitchAugment = 0, gridTempo = true } = {},
  modelCache = null,
) {
  const mono = toMono(audioBuffer);
  const x = resample(mono, audioBuffer.sampleRate, ANALYSIS_SR);

  // Prefer the neural transcriber; fall back to YIN if the model can't load
  // (offline, CDN blocked) or hears nothing.
  let notes = null;
  let engine = "basic-pitch";
  const cache = modelCache ?? {};
  try {
    const x22 = resample(mono, audioBuffer.sampleRate, BASIC_PITCH_SR);
    notes = await transcribeBasicPitch(x22, onProgress, sensitivity, cache, adaptivePitch);
    // Cross-validate every model note against the raw signal with YIN:
    // drops notes with no periodicity (breaths, consonants), octave-corrects
    // harmonic confusions, and refines exact pitch for tuning estimation.
    if (notes) {
      const before = notes.length;
      const yin = createYin(); // shared working buffers across all probes
      notes = notes.filter((n) => refineNote(x, n, yin));
      if (before > notes.length) {
        console.debug(`YIN validation dropped ${before - notes.length} of ${before} notes`);
      }
    }
  } catch (err) {
    console.warn("Basic Pitch unavailable, falling back to YIN:", err);
    notes = null;
  }
  if (!notes || notes.length === 0) {
    engine = "yin";
    const frames = await pitchTrack(x, onProgress);
    notes = segmentNotes(frames);
  }
  if (notes.length === 0) {
    throw new Error("Couldn't detect any sung notes. Try recording again, closer to the mic and with a clearer melody.");
  }

  // Test-time pitch augmentation. The model is measurably weaker below ~MIDI
  // 53, which is exactly where low voices live. Rather than compensate with
  // thresholds, transcribe the take a second time shifted UP into the range
  // where the model is strong, then map the notes back down. Speeding the
  // waveform up by r raises pitch by `pitchAugment` semitones and divides
  // every timestamp by r, so undoing it is a multiply and a subtract.
  // The extra pass only fills gaps — the unshifted pass stays authoritative.
  if (pitchAugment > 0 && engine === "basic-pitch") {
    try {
      const r = Math.pow(2, pitchAugment / 12);
      const xUp = resample(
        resample(mono, audioBuffer.sampleRate, BASIC_PITCH_SR),
        BASIC_PITCH_SR, BASIC_PITCH_SR / r,
      );
      cache.aug ??= {};
      // adaptivePitch off: the point of shifting is to leave the low-voice
      // regime, so re-applying the low-voice boost would double-compensate.
      const augRaw = await transcribeBasicPitch(xUp, () => {}, sensitivity, cache.aug, false);
      const mapped = (augRaw ?? []).map((n) => ({
        start: n.start * r,
        end: n.end * r,
        dur: (n.end - n.start) * r,
        midiFloat: n.midiFloat - pitchAugment,
        amp: n.amp,
        augmented: true,
      }));
      const fresh = mapped.filter((a) =>
        a.dur >= 0.08 && !notes.some((n) => Math.min(n.end, a.end) - Math.max(n.start, a.start) > 0));
      if (fresh.length) {
        const yin = createYin();
        const kept = fresh.filter((n) => refineNote(x, n, yin));
        if (kept.length) notes = [...notes, ...kept].sort((a, b) => a.start - b.start);
        console.debug(`Pitch augment (+${pitchAugment}): ${kept.length} of ${fresh.length} gap notes kept`);
      }
    } catch (err) {
      console.warn("Pitch augmentation unavailable:", err);
    }
  }

  // Second-pass recovery: re-read the cached posteriorgram in the gaps the
  // first pass left empty, then hold the candidates to the same YIN standard
  // as everything else — proposing is cheap, so the referee does the work.
  if (gapRecovery && engine === "basic-pitch" && cache.frames?.length) {
    const opts = typeof gapRecovery === "object" ? gapRecovery : {};
    const cands = recoverGapNotes(cache, notes, audioBuffer.duration, opts);
    if (cands.length) {
      const yin = createYin();
      const kept = cands.filter((n) => refineNote(x, n, yin));
      if (kept.length) {
        notes = [...notes, ...kept].sort((a, b) => a.start - b.start);
      }
      console.debug(`Gap recovery: ${kept.length} of ${cands.length} candidates kept`);
    }
  }

  // Octave correction via sub-harmonic summation (opt-in, default off).
  // Measured inert on Vocadito: the YIN cross-validation above already fixes
  // octaves on real voices, and low-voice failures turned out to be onset
  // *detection*, not pitch. Kept for extreme low-fundamental input (bass /
  // instruments) where YIN's autocorrelation can octave-slip.
  if (octaveFix) {
    const fixed = octaveCorrect(x, notes);
    if (fixed) console.debug(`Octave correction shifted ${fixed} of ${notes.length} notes`);
  }

  // Rejoin held notes the model split into same-pitch fragments — merge only
  // where the audio shows continuous energy across the junction, so genuine
  // re-articulated repeats (which dip at the closure) survive. Runs before
  // onset snap so snapping only ever sees real onsets.
  if (consolidate) {
    const res = consolidateRepeats(x, notes);
    notes = res.notes;
    if (res.merged) console.debug(`Consolidation merged ${res.merged} held-note fragments`);
  }

  // Energy moves at the moment of articulation, faster than pitch
  // stabilizes — snap note starts to nearby energy rises for sharper
  // timing (better beat grid, better export quantization).
  if (snapOnsets) {
    const snapped = snapNoteOnsets(x, notes);
    console.debug(`Onset snap adjusted ${snapped} of ${notes.length} note starts`);
  }

  // The same idea applied to note ends — MEASURED HARMFUL, hence default off.
  // On Vocadito it cost 4.1 points of COnPOff, the very metric it targets,
  // because a held note's energy dips mid-note (vibrato, tremolo) and the
  // decay search truncates there; shortened notes then fall under the
  // minimum-length filter and disappear entirely (up to 10 lost on a clip).
  // Kept opt-in for material with harder, more percussive note ends.
  if (snapOffsets) {
    const snapped = snapNoteOffsets(x, notes);
    console.debug(`Offset snap adjusted ${snapped} of ${notes.length} note ends`);
  }

  // Syllable lock: when the singer articulates every note with the same
  // syllable ("Doo"), learn that attack signature from the take and use it
  // to reject non-notes and to settle split-vs-restrike questions.
  let lockInfo = { enabled: syllableLock, engaged: false, dropped: 0, merged: 0 };
  if (syllableLock) {
    lockInfo = applySyllableLock(x, notes);
    notes = lockInfo.notes;
    console.debug(`Syllable lock: ${lockInfo.engaged ? `engaged, dropped ${lockInfo.dropped}, merged ${lockInfo.merged}` : `not engaged (${lockInfo.reason})`}`);
  }

  // Tempo pre-estimate so the minimum note length can be expressed in
  // musical units ("nothing shorter than a 16th") before the final grid.
  let { tempo, beats } = detectBeats(x, notes, audioBuffer.duration);

  if (minNote > 0 && notes.length > 1) {
    const res = applyMinNote(notes, 0.7 * minNote * (60 / tempo));
    notes = res.notes;
    if (res.absorbed + res.dropped > 0) {
      console.debug(`Min note: absorbed ${res.absorbed}, dropped ${res.dropped}`);
    }
  }
  if (notes.length === 0) {
    throw new Error("All detected notes were filtered out — try a higher sensitivity or a longer minimum note setting.");
  }

  // Singers are rarely centered on A=440. Estimate the global offset, round
  // pitches relative to it (better note/key accuracy), and report it so the
  // backing can be detuned to match the voice.
  const tuningCents = estimateTuning(notes);
  for (const n of notes) n.midi = Math.round(n.midiFloat - tuningCents / 100);

  const key = detectKey(notes);

  // Auto-tune: simplify the melody to its skeleton before arranging —
  // ornament notes are absorbed into their neighbors and brief out-of-key
  // notes snap to the scale. Expressive detail is what the singer adds;
  // the accompaniment wants the underlying tune. Intensity 0..1 scales how
  // aggressive the simplification is (legacy `true` means 0.5).
  const autoTuneIntensity = autoTune === true ? 0.5 : Number(autoTune) || 0;
  if (autoTuneIntensity > 0) {
    const before = notes.length;
    notes = simplifyNotes(notes, key, autoTuneIntensity);
    console.debug(`Auto-tune (${Math.round(autoTuneIntensity * 100)}%) simplified ${before} notes to ${notes.length}`);
  }

  // Final grid, built only after every cleaning stage has had its say —
  // junk notes must not become grid anchors or onset-envelope spikes.
  ({ tempo, beats } = detectBeats(x, notes, audioBuffer.duration));

  // Note-anchored grid: classify each inter-onset interval as a musical
  // duration and re-anchor the grid at every note, so a wandering internal
  // tempo can't accumulate timing error. The energy-tracked grid above
  // still provides the initial tempo estimate (and the fallback).
  if (followNotes && notes.length >= 4) {
    beats = noteAnchoredGrid(notes, tempo, audioBuffer.duration);
    const intervals = [];
    for (let i = 1; i < beats.length; i++) intervals.push(beats[i] - beats[i - 1]);
    intervals.sort((a, b) => a - b);
    // The note-anchored grid drifts to eighth-note spacing on syllabic singing;
    // fold the tempo and thin the grid so bars span a real 4/4 bar.
    const medianBpm = 60 / intervals[Math.floor(intervals.length / 2)];
    // gridTempo: pick the pulse the onsets actually sit on rather than their
    // commonest spacing (see rhythm.bestGridTempo).
    ({ tempo, beats } = foldTempo(gridTempo ? bestGridTempo(notes) : medianBpm, beats));
  }

  return {
    notes,
    key,
    tempo,
    beats,
    tuningCents,
    engine,
    backend: engine === "basic-pitch" ? getBackend() : null,
    lockInfo,
    modelCache: cache,
    firstOnset: notes[0].start,
    duration: audioBuffer.duration,
  };
}

// ---------------------------------------------------------------------------
// YIN fallback engine (used only when the neural model can't load)
// ---------------------------------------------------------------------------

async function pitchTrack(x, onProgress) {
  const yin = createYin();
  const nFrames = Math.max(0, Math.floor((x.length - FRAME_SIZE) / HOP));
  const frames = [];

  for (let i = 0; i < nFrames; i++) {
    frames.push(yinFrame(x, i * HOP, yin));
    if (i % 150 === 149) {
      onProgress(i / nFrames);
      await tick();
    }
  }
  onProgress(1);

  // Median-filter the midi track (window 5) to kill octave blips.
  const midi = frames.map((f) => f.midi);
  for (let i = 0; i < frames.length; i++) {
    const win = [];
    for (let j = Math.max(0, i - 2); j <= Math.min(frames.length - 1, i + 2); j++) {
      if (frames[j].voiced) win.push(midi[j]);
    }
    if (win.length >= 3 && frames[i].voiced) frames[i].midi = median(win);
  }
  return frames;
}

function segmentNotes(frames) {
  const frameDur = HOP / ANALYSIS_SR;
  const notes = [];
  let cur = null; // { startIdx, midis: [], gap }

  const closeNote = (endIdx) => {
    if (!cur) return;
    if (cur.midis.length >= MIN_NOTE_FRAMES) {
      notes.push({
        start: frames[cur.startIdx].t,
        end: frames[cur.startIdx].t + (endIdx - cur.startIdx) * frameDur,
        midiFloat: median(cur.midis), // rounded to .midi after tuning estimation
      });
    }
    cur = null;
  };

  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    if (f.voiced) {
      if (cur) {
        if (Math.abs(f.midi - median(cur.midis)) > NOTE_SPLIT_SEMITONES) {
          closeNote(i);
          cur = { startIdx: i, midis: [f.midi], gap: 0 };
        } else {
          cur.midis.push(f.midi);
          cur.gap = 0;
        }
      } else {
        cur = { startIdx: i, midis: [f.midi], gap: 0 };
      }
    } else if (cur) {
      cur.gap++;
      if (cur.gap > MAX_UNVOICED_GAP) closeNote(i - cur.gap + 1);
    }
  }
  closeNote(frames.length);

  return notes.map((n) => ({ ...n, dur: n.end - n.start }));
}
