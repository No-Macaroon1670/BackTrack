// Accuracy evaluation harness. Runs analyzeVocal over a vocal dataset and
// scores the output against reference annotations with mir_eval-style
// metrics (metrics.js). Standalone from the main app (loaded by eval.html)
// but reuses the exact analysis pipeline.
//
// Two entry points:
//   - runDataset(): manual file-picker path (any supported format).
//   - runManifest() / runComparison(): auto-load a served dataset from a
//     manifest.json (see tools/build-eval-manifest.ps1) and, for compare,
//     sweep several analysis configs reusing one model inference per clip.
//
// Annotation formats, matched to audio by basename:
//   - F0 CSV/TSV: "time,freq" per line (Vocadito f0, MIREX) — frame level
//   - Note CSV:   Vocadito "start,pitchHz,duration" or "start,end,pitch"
//   - .pv:        one semitone value per 10 ms line (MIR-1K) — frame level
//
// Datasets are large and licensed, so nothing ships in the repo.

import { analyzeVocal } from "./analyze.js";
import { mixToWavBlob } from "./render.js";
import {
  noteTranscriptionMetrics, melodyMetrics, notesToF0, midiToHz, hzToMidi, mean,
} from "./metrics.js";

const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

// ---------------------------------------------------------------------------
// Annotation parsing
// ---------------------------------------------------------------------------

function splitRows(text) {
  return text.trim().split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !/^[a-zA-Z#]/.test(l));
}
function fields(line) {
  return line.split(/[,\t; ]+/).map(Number);
}
function median(a) {
  const s = [...a].sort((x, y) => x - y);
  return s[s.length >> 1];
}

/** Frame F0 from "time,freq" rows; freq <= 0 means unvoiced. */
export function parseF0Csv(text) {
  const times = [], hz = [];
  for (const line of splitRows(text)) {
    const f = fields(line);
    if (f.length < 2 || Number.isNaN(f[0])) continue;
    times.push(f[0]);
    hz.push(f[1] > 0 ? f[1] : 0);
  }
  return { kind: "f0", times, hz };
}

/** MIR-1K .pv: one MIDI-semitone value per 10 ms frame; 0 = unvoiced. */
export function parsePv(text, hop = 0.01) {
  const times = [], hz = [];
  splitRows(text).forEach((line, i) => {
    const v = Number(line.split(/[,\t ]+/)[0]);
    times.push(i * hop);
    hz.push(v > 0 ? midiToHz(v) : 0);
  });
  return { kind: "f0", times, hz };
}

/**
 * Note list from 3-column rows. `format`:
 *   "vocadito" → start_sec, pitch_Hz, duration_sec
 *   "auto"     → detect start,end,pitch vs start,pitchHz,duration; pitch
 *                column auto-detected MIDI vs Hz.
 */
export function parseNoteCsv(text, format = "auto") {
  const rows = splitRows(text).map(fields).filter((f) => f.length >= 3 && !Number.isNaN(f[0]));
  let notes;
  if (format === "vocadito") {
    notes = rows.map((f) => ({ onset: f[0], offset: f[0] + f[2], midi: hzToMidi(f[1]) }));
  } else {
    // Distinguish start,end,pitch from start,pitch,duration: in the former
    // col2 (end) exceeds col1 (start) for essentially every row.
    const endLike = rows.filter((f) => f[1] > f[0]).length > rows.length * 0.8;
    if (endLike) {
      const asHz = median(rows.map((f) => f[2])) > 128;
      notes = rows.map((f) => ({ onset: f[0], offset: f[1], midi: asHz ? hzToMidi(f[2]) : f[2] }));
    } else {
      const asHz = median(rows.map((f) => f[1])) > 128;
      notes = rows.map((f) => ({ onset: f[0], offset: f[0] + f[2], midi: asHz ? hzToMidi(f[1]) : f[1] }));
    }
  }
  return { kind: "notes", notes };
}

export function parseAnnotation(filename, text, noteFormat = "auto") {
  const ext = filename.slice(filename.lastIndexOf(".")).toLowerCase();
  if (ext === ".pv") return parsePv(text);
  const first = splitRows(text)[0];
  if (first && fields(first).filter((v) => !Number.isNaN(v)).length >= 3) return parseNoteCsv(text, noteFormat);
  return parseF0Csv(text);
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/** Scores one analysis against whichever annotations are present. */
function scoreAnalysis(analysis, { f0, notes }) {
  const est = analysis.notes.map((n) => ({ onset: n.start, offset: n.end, midi: n.midi }));
  const r = { engine: analysis.engine, nEst: est.length };
  if (notes) r.note = noteTranscriptionMetrics(notes.notes, est);
  if (f0) r.frame = melodyMetrics(f0.hz, notesToF0(analysis.notes, f0.times));
  return r;
}

async function decode(arrayBuffer) {
  return audioCtx.decodeAudioData(arrayBuffer.slice(0)); // decode detaches; hand it a copy
}

/** Analyzes one decoded clip and scores it. `annos` = { f0?, notes? }. */
export async function runClip(buffer, annos, settings = {}, cache = null) {
  const analysis = await analyzeVocal(buffer, () => {}, settings, cache);
  return { result: scoreAnalysis(analysis, annos), cache: analysis.modelCache };
}

function aggregate(rows) {
  const ok = rows.filter((r) => !r.error);
  const noteRows = ok.filter((r) => r.note);
  const frameRows = ok.filter((r) => r.frame);
  const summary = { clips: ok.length, failed: rows.length - ok.length };
  if (noteRows.length) {
    summary.note = {
      onsetF: mean(noteRows.map((r) => r.note.onset.f)),
      onpF: mean(noteRows.map((r) => r.note.onp.f)),
      onpoffF: mean(noteRows.map((r) => r.note.onpoff.f)),
    };
  }
  if (frameRows.length) {
    summary.frame = {
      rpa: mean(frameRows.map((r) => r.frame.rpa)),
      rca: mean(frameRows.map((r) => r.frame.rca)),
      voicingRecall: mean(frameRows.map((r) => r.frame.voicingRecall)),
      voicingFalseAlarm: mean(frameRows.map((r) => r.frame.voicingFalseAlarm)),
    };
  }
  return summary;
}

// ---------------------------------------------------------------------------
// Manual file-picker path
// ---------------------------------------------------------------------------

/** @param {Array<{name, audio: ArrayBuffer, annText, annName}>} clips */
export async function runDataset(clips, settings, onProgress = () => {}) {
  const rows = [];
  for (let i = 0; i < clips.length; i++) {
    const clip = clips[i];
    onProgress(i, clips.length, clip.name);
    try {
      const buffer = await decode(clip.audio);
      const ann = parseAnnotation(clip.annName, clip.annText);
      const annos = ann.kind === "notes" ? { notes: ann } : { f0: ann };
      const { result } = await runClip(buffer, annos, settings);
      rows.push({ name: clip.name, ...result });
    } catch (err) {
      rows.push({ name: clip.name, error: String(err.message || err) });
    }
  }
  onProgress(clips.length, clips.length, "");
  return { rows, summary: aggregate(rows) };
}

// ---------------------------------------------------------------------------
// Auto path: served dataset via manifest.json
// ---------------------------------------------------------------------------

export async function loadManifest(url = "vocadito/manifest.json") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`manifest not found at ${url} (HTTP ${res.status})`);
  return res.json();
}

async function fetchClip(entry, noteFormat) {
  const buffer = await decode(await (await fetch(entry.audio)).arrayBuffer());
  const annos = {};
  if (entry.f0) annos.f0 = parseF0Csv(await (await fetch(entry.f0)).text());
  if (entry.notesA1) annos.notes = parseNoteCsv(await (await fetch(entry.notesA1)).text(), noteFormat);
  return { buffer, annos };
}

/** Auto-load + score a served dataset. `settings` applies to every clip. */
export async function runManifest(manifest, settings = {}, onProgress = () => {}, max = Infinity) {
  const clips = manifest.clips.slice(0, max);
  const rows = [];
  for (let i = 0; i < clips.length; i++) {
    onProgress(i, clips.length, clips[i].id);
    try {
      const { buffer, annos } = await fetchClip(clips[i], manifest.noteFormat);
      const { result } = await runClip(buffer, annos, settings);
      rows.push({ name: clips[i].id, ...result });
    } catch (err) {
      rows.push({ name: clips[i].id, error: String(err.message || err) });
    }
  }
  onProgress(clips.length, clips.length, "");
  return { rows, summary: aggregate(rows) };
}

/** Default config sweep for the compare view. */
export const COMPARE_CONFIGS = [
  { name: "default", settings: {} },
  { name: "no adaptive", settings: { adaptivePitch: false } },
  { name: "no consolidate", settings: { consolidate: false } },
  { name: "sens 30", settings: { sensitivity: 0.3 } },
  { name: "sens 70", settings: { sensitivity: 0.7 } },
  { name: "no snap", settings: { snapOnsets: false } },
  { name: "no min-note", settings: { minNote: 0 } },
];

/**
 * Runs every config over the dataset, reusing one model inference per clip
 * (settings only affect post-processing), so a 6-config sweep costs barely
 * more than a single run. @returns per-config summaries.
 */
export async function runComparison(manifest, configs = COMPARE_CONFIGS, onProgress = () => {}, max = Infinity) {
  const clips = manifest.clips.slice(0, max);
  const perConfigRows = configs.map(() => []);
  for (let i = 0; i < clips.length; i++) {
    onProgress(i, clips.length, clips[i].id);
    try {
      const { buffer, annos } = await fetchClip(clips[i], manifest.noteFormat);
      const cache = {}; // one inference, reused across configs
      for (let c = 0; c < configs.length; c++) {
        const { result } = await runClip(buffer, annos, configs[c].settings, cache);
        perConfigRows[c].push({ name: clips[i].id, ...result });
      }
    } catch (err) {
      for (const rows of perConfigRows) rows.push({ name: clips[i].id, error: String(err.message || err) });
    }
  }
  onProgress(clips.length, clips.length, "");
  return configs.map((cfg, c) => ({ name: cfg.name, summary: aggregate(perConfigRows[c]) }));
}

// ---------------------------------------------------------------------------
// Self-test: proves metrics + the full path on synthetic ground truth
// ---------------------------------------------------------------------------

export async function selfTest() {
  const results = {};
  const ref = [
    { onset: 0.0, offset: 0.5, midi: 60 },
    { onset: 1.0, offset: 1.5, midi: 64 },
    { onset: 2.0, offset: 2.6, midi: 67 },
  ];
  results.perfect = noteTranscriptionMetrics(ref, ref).onpoff.f;
  results.missOneRecall = noteTranscriptionMetrics(ref, ref.slice(0, 2)).onp.recall;
  const wrongPitch = noteTranscriptionMetrics(ref, ref.map((r) => ({ ...r, midi: r.midi + 1 })));
  results.wrongPitchOnsetF = wrongPitch.onset.f;
  results.wrongPitchOnpF = wrongPitch.onp.f;

  const grid = Array.from({ length: 100 }, () => 440);
  results.frameSame = melodyMetrics(grid, grid).rpa;
  results.frameOctave = melodyMetrics(grid, grid.map(() => 880));
  results.frameOctaveRpa = results.frameOctave.rpa;
  results.frameOctaveRca = results.frameOctave.rca;

  // Vocadito note-format parse round-trip.
  const parsed = parseNoteCsv("0.45,104.585,0.081\n0.54,126.418,0.157", "vocadito");
  results.vocaditoParse = parsed.notes.length === 2 &&
    Math.abs(parsed.notes[0].offset - 0.531) < 1e-6 &&
    Math.abs(parsed.notes[0].midi - hzToMidi(104.585)) < 1e-6;

  // End-to-end: synth melody → WAV → decode → analyze → score.
  const sr = 44100, notes = [];
  const midis = [60, 62, 64, 65, 67, 69, 71, 72];
  const buf = new AudioBuffer({ length: Math.ceil((midis.length * 0.6 + 1) * sr), sampleRate: sr, numberOfChannels: 1 });
  const d = buf.getChannelData(0);
  midis.forEach((m, i) => {
    const start = 0.3 + i * 0.6, end = start + 0.5;
    notes.push({ onset: start, offset: end, midi: m });
    const f = midiToHz(m), s0 = Math.floor(start * sr), s1 = Math.floor(end * sr);
    for (let j = s0; j < s1; j++) {
      const env = Math.min(1, (j - s0) / 500) * Math.min(1, (s1 - j) / 800);
      const ph = 2 * Math.PI * f * (j - s0) / sr;
      d[j] = env * (0.32 * Math.sin(ph) + 0.13 * Math.sin(2 * ph));
    }
  });
  const decoded = await decode(await mixToWavBlob(null, buf, 0, 1).arrayBuffer());
  const { result: e2e } = await runClip(decoded, { notes: { notes } }, {});
  results.e2eOnsetF = e2e.note.onset.f;
  results.e2eOnpF = e2e.note.onp.f;
  results.e2eNEst = e2e.nEst;
  return results;
}
