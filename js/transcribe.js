// Neural note transcription via Spotify's Basic Pitch (TensorFlow.js build),
// loaded from a CDN at first use and run entirely in the browser. analyze.js
// falls back to the built-in YIN tracker when this is unavailable.
//
// This module owns model loading, backend selection (GPU inline vs. CPU
// worker), and turning raw model output into a note list. The note-hygiene
// heuristics applied to that list live in notes.js.

import { cleanNotes } from "./notes.js";

// Re-exported so cleanNotes remains importable from this module (its historic home).
export { cleanNotes };

/**
 * Extracts EVERY pitch the model considered plausible from the cached
 * posteriorgram — not just the winning monophonic line — as notes carrying
 * their activation as `prob` (0..1). Per pitch bin, threshold-crossing runs
 * become notes; prob is the run's mean activation. Used by the
 * probability-layers MIDI export, where prob maps to velocity: the melody
 * comes out loud, octave/harmonic candidates quiet, ambiguity as soft
 * texture.
 *
 * @returns {Array<{start,end,midi,prob,peak}>|null} null without model output
 *   (YIN fallback or MIDI input have no posteriorgram).
 */
export function extractProbabilityNotes(cache, { threshold = 0.15, minDur = 0.07 } = {}) {
  const frames = cache?.frames;
  if (!frames?.length || frames[0].length !== 88) return null;
  const hopSec = 256 / BASIC_PITCH_SR; // the model's frame hop
  const minFrames = Math.max(2, Math.round(minDur / hopSec));
  const notes = [];
  for (let c = 0; c < 88; c++) {
    let run = -1, sum = 0, peak = 0;
    for (let t = 0; t <= frames.length; t++) {
      const v = t < frames.length ? frames[t][c] : 0;
      if (v >= threshold) {
        if (run < 0) { run = t; sum = 0; peak = 0; }
        sum += v;
        if (v > peak) peak = v;
      } else if (run >= 0) {
        const len = t - run;
        if (len >= minFrames) {
          notes.push({ start: modelFrameToTime(run), end: modelFrameToTime(t), midi: c + 21, prob: sum / len, peak });
        }
        run = -1;
      }
    }
  }
  notes.sort((a, b) => a.start - b.start);
  return notes;
}

/**
 * Frame index -> seconds, mirroring the library's modelFrameToTime (toMidi.js).
 * Inference runs in 2 s windows of 172 frames whose concatenation overcounts
 * time by ~10.3 ms per window, so plain frame*hop drifts audibly late against
 * the melody notes (which the library converts with this correction applied).
 */
export function modelFrameToTime(frame) {
  const hop = 256 / BASIC_PITCH_SR;
  const annotFrames = Math.floor(BASIC_PITCH_SR / 256) * 2; // 172
  const windowOffset = hop * (annotFrames - (2 * BASIC_PITCH_SR - 256) / 256) + 0.0018;
  return frame * hop - windowOffset * Math.floor(frame / annotFrames);
}

/** Seconds -> nearest frame index; approximate inverse of modelFrameToTime. */
function modelTimeToFrame(t) {
  const hop = 256 / BASIC_PITCH_SR;
  const annotFrames = Math.floor(BASIC_PITCH_SR / 256) * 2;
  const windowOffset = hop * (annotFrames - (2 * BASIC_PITCH_SR - 256) / 256) + 0.0018;
  const winSec = annotFrames * hop - windowOffset;
  const w = Math.max(0, Math.floor(t / winSec));
  return Math.round((t + windowOffset * w) / hop);
}

/**
 * Stamps each note with `conf` (0..1): how strongly the model's posteriorgram
 * supports the note AS DRAWN — peak activation over its current span at its
 * current pitch (±1 semitone bin, tolerating tuning offsets). Recomputed
 * after edits, so a note dragged to a pitch the model never heard fades out
 * honestly. Returns false (notes untouched) without model output.
 */
export function noteConfidence(cache, notes) {
  const frames = cache?.frames;
  if (!frames?.length || frames[0].length !== 88) return false;
  for (const n of notes) {
    const bin = Math.round(n.midiFloat ?? n.midi) - 21;
    const f0 = Math.max(0, modelTimeToFrame(n.start) - 2);
    const f1 = Math.min(frames.length - 1, modelTimeToFrame(n.end) + 2);
    let peakAct = 0;
    for (let f = f0; f <= f1; f++) {
      for (let b = Math.max(0, bin - 1); b <= Math.min(87, bin + 1); b++) {
        if (frames[f][b] > peakAct) peakAct = frames[f][b];
      }
    }
    // Solid detections peak ~0.6-0.95; below ~0.1 the model heard nothing.
    n.conf = Math.max(0, Math.min(1, (peakAct - 0.1) / 0.7));
  }
  return true;
}

/**
 * Activation-weighted mean pitch (MIDI) of the note posteriorgram, used to
 * detect low-voiced takes. Only bins above a floor count, so broadband noise
 * doesn't pull the estimate. `frames` is [time][88], MIDI 21-108.
 */
function frameCentroidMidi(frames) {
  let num = 0, den = 0;
  for (const f of frames) {
    for (let c = 0; c < 88; c++) {
      if (f[c] > 0.3) { num += (c + 21) * f[c]; den += f[c]; }
    }
  }
  return den > 0 ? num / den : 60;
}

const BP_VERSION = "1.0.1";
const TF_VERSION = "3.21.0"; // matches basic-pitch 1.0.1's @tensorflow/tfjs ^3.2.0
const MODEL_URL = `https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@${BP_VERSION}/model/model.json`;
// Self-contained bundle (CPU) for the Worker fallback; the ?deps build lets us
// share an explicit WebGL-backed TF.js instance on the GPU path.
const LIB_BUNDLE_URL = `https://esm.sh/@spotify/basic-pitch@${BP_VERSION}?bundle`;
const LIB_DEPS_URL = `https://esm.sh/@spotify/basic-pitch@${BP_VERSION}?deps=@tensorflow/tfjs@${TF_VERSION}`;
const TF_URL = `https://esm.sh/@tensorflow/tfjs@${TF_VERSION}`;

export const BASIC_PITCH_SR = 22050; // the model's expected sample rate

// The backend the last analysis actually ran on ("webgl" = GPU, "cpu"), so the
// UI can show which path engaged. Set when the lib + backend are resolved.
let activeBackend = null;
export function getBackend() {
  return activeBackend;
}

// Real (hardware) GPU backing WebGL? Software WebGL (SwiftShader/llvmpipe, seen
// in headless/VM browsers) is slower than CPU for this model AND janky on the
// GPU path, so we treat it as "no GPU" and use the CPU Worker instead.
let _hwGL = null;
function hardwareWebGL() {
  if (_hwGL === null) {
    try {
      const gl = document.createElement("canvas").getContext("webgl");
      const ext = gl && gl.getExtension("WEBGL_debug_renderer_info");
      const r = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : "";
      _hwGL = !!r && !/swiftshader|software|llvmpipe|basic render|microsoft basic/i.test(r);
    } catch (e) {
      _hwGL = false;
    }
  }
  return _hwGL;
}

// A network that black-holes the CDN (captive portals, strict proxies, some
// national routes) never rejects — it just hangs, which would leave the user
// staring at a stalled progress bar forever. Every remote wait is therefore
// bounded: on timeout the promise rejects, analyze.js catches it, and the
// built-in YIN tracker takes over. The slow request may still finish in the
// background and warm the HTTP cache for the next attempt.
const LIB_TIMEOUT_MS = 25000;

function withTimeout(promise, ms, what) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`${what} timed out after ${Math.round(ms / 1000)}s`)),
      ms,
    );
    promise.then(
      (v) => { clearTimeout(timer); resolve(v); },
      (e) => { clearTimeout(timer); reject(e); },
    );
  });
}

let libPromise = null;
function loadLib() {
  if (!libPromise) {
    libPromise = withTimeout((async () => {
      if (hardwareWebGL()) {
        // GPU path: load TF.js explicitly, switch it to WebGL, and hand that
        // same instance to Basic Pitch (?deps pins both to one TF.js).
        const tf = await import(/* webpackIgnore: true */ TF_URL);
        try { await tf.setBackend("webgl"); } catch (e) { /* fall back to default */ }
        await tf.ready();
        activeBackend = tf.getBackend();
        console.debug(`Basic Pitch backend: ${activeBackend} (main thread)`);
        return import(/* webpackIgnore: true */ LIB_DEPS_URL);
      }
      activeBackend = "cpu";
      console.debug("Basic Pitch backend: cpu (worker)");
      return import(/* webpackIgnore: true */ LIB_BUNDLE_URL);
    })(), LIB_TIMEOUT_MS, "Model library download");
    // A rejected promise is still truthy, so without this a single flaky CDN
    // load would pin the whole session to the YIN fallback with no way back.
    // Clearing the cache lets the next take retry (as sampler.js does).
    libPromise.catch(() => { libPromise = null; });
  }
  return libPromise;
}

// One shared BasicPitch instance for the inline (GPU) path: constructing it
// starts the graph-model download/init, and reusing it across takes skips
// that cost on every new recording. The constructor kicks off that download
// and stores it as a promise, so a failed download would otherwise be cached
// forever too — drop the instance on rejection.
let bpInstance = null;
function getBP(BasicPitch) {
  if (!bpInstance) {
    bpInstance = new BasicPitch(MODEL_URL);
    Promise.resolve(bpInstance.model).catch(() => { bpInstance = null; });
  }
  return bpInstance;
}

/**
 * Pre-warms the transcriber: fetches the TF.js + Basic Pitch libraries and
 * downloads the model weights, so the first analysis doesn't stall on several
 * MB of network. app.js calls this on the user's first sign of intent (not at
 * page load) — a visitor who only reads the page should not pay for a model
 * they never use.
 *
 * Resolves true only once the weights are actually usable, because the "still
 * downloading" phase text keys off it; never throws.
 *
 * @returns {Promise<boolean>}
 */
export function warmup() {
  return loadLib()
    .then(async ({ BasicPitch }) => {
      if (hardwareWebGL()) {
        await getBP(BasicPitch).model; // the weights, not just the library
      } else {
        await fetch(MODEL_URL); // prime the HTTP cache for the worker
      }
      return true;
    })
    .catch(() => false);
}

// Run the model's inference where it's fastest while keeping the UI alive.
// With a real GPU, WebGL inference runs inline: it's ~3x faster than CPU and
// yields between GPU dispatches (worst-case main-thread stall ~1s, far under
// the "page unresponsive" threshold). Without a GPU, CPU inference must run in
// a Worker or it freezes the page for the whole analysis.
function runInference(x, onProgress, BasicPitch) {
  const inline = () => {
    const frames = [], onsets = [], contours = [];
    const bp = getBP(BasicPitch);
    return bp
      .evaluateModel(
        x,
        (f, o, c) => { frames.push(...f); onsets.push(...o); contours.push(...c); },
        (p) => onProgress(p),
      )
      .then(() => ({ frames, onsets, contours }));
  };
  // The model weights download lazily inside the first inference, so this
  // budget covers network as well as compute: generous per second of audio,
  // with a floor for short takes.
  const budget = Math.max(60000, Math.round((x.length / BASIC_PITCH_SR) * 4000));
  if (hardwareWebGL()) return withTimeout(inline(), budget, "Inference");
  return withTimeout(runInferenceInWorker(x, onProgress, budget), budget, "Inference")
    .catch((err) => {
      console.warn("Inference worker unavailable, running on main thread:", err);
      return withTimeout(inline(), budget, "Inference");
    });
}

function runInferenceInWorker(x, onProgress, budget) {
  return new Promise((resolve, reject) => {
    let worker;
    try {
      worker = new Worker(new URL("./transcribe.worker.js", import.meta.url), { type: "module" });
    } catch (e) {
      reject(e);
      return;
    }
    // Kill a wedged worker rather than leaking it when the outer timeout fires.
    const killer = setTimeout(() => {
      try { worker.terminate(); } catch { /* already gone */ }
      reject(new Error("Inference worker timed out"));
    }, budget);
    const settle = (fn) => (v) => { clearTimeout(killer); fn(v); };
    resolve = settle(resolve);
    reject = settle(reject);
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "backend") {
        console.debug(`Basic Pitch inference backend: ${m.backend}`);
      } else if (m.type === "progress") {
        onProgress(m.p);
      } else if (m.type === "done") {
        worker.terminate();
        resolve({ frames: m.frames, onsets: m.onsets, contours: m.contours });
      } else if (m.type === "error") {
        worker.terminate();
        reject(new Error(m.message));
      }
    };
    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(e.message || "worker error"));
    };
    // Hand the audio off by transfer (copy first so we don't neuter the caller's buffer).
    const copy = x.slice();
    worker.postMessage({ type: "infer", audio: copy }, [copy.buffer]);
  });
}

/**
 * @param {Float32Array} x audio resampled to BASIC_PITCH_SR, mono
 * @param {(p: number) => void} onProgress 0..1
 * @param {number} sensitivity 0..1; low = keep only loud, confident notes,
 *   high = catch quiet notes too. 0.5 reproduces the original defaults.
 * @param {object|null} cache model-output cache. Inference depends only on
 *   the audio, never on settings — pass the same object across re-analyses
 *   of one take and the slow model step is skipped entirely.
 * @returns {Promise<Array<{start, end, dur, midiFloat, amp}>>} monophonic note list
 */
export async function transcribeBasicPitch(x, onProgress = () => {}, sensitivity = 0.5, cache = null, adaptivePitch = true) {
  const { BasicPitch, noteFramesToTime, addPitchBendsToNoteEvents, outputToNotesPoly } =
    await loadLib();

  let frames, onsets, contours;
  if (cache && cache.frames) {
    ({ frames, onsets, contours } = cache);
    onProgress(1);
  } else {
    ({ frames, onsets, contours } = await runInference(x, onProgress, BasicPitch));
    if (cache) Object.assign(cache, { frames, onsets, contours });
  }

  // Low voices have weak onset salience, so the fixed threshold under-detects
  // them (measured on Vocadito: low-pitch clips gain both recall AND precision
  // from a lower threshold, while high voices don't). Read the pitch region
  // straight from the cached posteriorgram and boost sensitivity for low
  // takes only — adaptive, no re-inference.
  let effSensitivity = sensitivity;
  if (adaptivePitch) {
    const centroid = frameCentroidMidi(frames);
    const boost = Math.max(0, Math.min(0.3, (57 - centroid) / 40));
    effSensitivity = Math.min(1, sensitivity + boost);
    if (boost > 0.02) {
      console.debug(`Adaptive pitch: centroid ${centroid.toFixed(1)} → sensitivity ${sensitivity.toFixed(2)}→${effSensitivity.toFixed(2)}`);
    }
  }

  // Sensitivity maps to the model's note-extraction thresholds (0.5 -> the
  // library defaults of onset 0.5 / frame 0.3), min length 8 frames
  // (~93 ms), inferred onsets, constrained to the vocal range 60-1000 Hz.
  const onsetThresh = 0.3 + (1 - effSensitivity) * 0.4;
  const frameThresh = 0.15 + (1 - effSensitivity) * 0.3;
  let rawEvents = outputToNotesPoly(frames, onsets, onsetThresh, frameThresh, 8, true, 1000, 60);

  // Peak-confidence gate: a real note spikes confidently at some point in
  // its life; threshold-flicker fragments never do. Uses the model's raw
  // frame activations (88 piano-key columns, MIDI 21-108).
  if (rawEvents.length && typeof rawEvents[0].startFrame === "number" &&
      frames.length && frames[0].length === 88) {
    const peakThresh = Math.min(0.85, frameThresh + 0.25);
    const before = rawEvents.length;
    rawEvents = rawEvents.filter((e) => {
      const col = e.pitchMidi - 21;
      if (col < 0 || col >= 88) return true;
      let peak = 0;
      const end = Math.min(frames.length, e.startFrame + e.durationFrames);
      for (let r = e.startFrame; r < end; r++) {
        if (frames[r][col] > peak) peak = frames[r][col];
      }
      return peak >= peakThresh;
    });
    if (before > rawEvents.length) {
      console.debug(`Peak-confidence gate dropped ${before - rawEvents.length} of ${before} events`);
    }
  }

  const events = noteFramesToTime(addPitchBendsToNoteEvents(contours, rawEvents));

  // Basic Pitch quantizes pitch to 1/3-semitone bins, too coarse for tuning
  // estimation — midiFloat is refined from the raw audio in analyze.js.
  const notes = events
    .map((e) => ({
      start: e.startTimeSeconds,
      end: e.startTimeSeconds + e.durationSeconds,
      dur: e.durationSeconds,
      amp: e.amplitude,
      midiFloat: e.pitchMidi,
    }))
    .sort((a, b) => a.start - b.start);

  // The amplitude gate scales with sensitivity too: strict keeps only notes
  // near the take's median loudness (hum the melody louder than the rest).
  return cleanNotes(notes, 0.25 + (1 - sensitivity) * 0.26);
}

