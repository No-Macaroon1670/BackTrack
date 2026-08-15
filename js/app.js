// UI wiring: connects recording/upload, analysis settings, generation, the
// player panel, and exports. Playback lives in player.js; canvas/DOM
// rendering in views.js; everything audio-analytical behind analyzeVocal.

import { Recorder } from "./recorder.js";
import { analyzeVocal, detectKey } from "./analyze.js";
import { harmonicStack, voiceTimbre } from "./notes.js";
import { toMono, resample, ANALYSIS_SR } from "./dsp.js";
import { warmup, extractProbabilityNotes, noteConfidence } from "./transcribe.js";
import { buildArrangement, keyName, uniformBeats } from "./arrange.js";
import { renderBacking, mixToWavBlob, renderMelodyPreview, renderLayers } from "./render.js";
import { loadVoiceSamples } from "./sampler.js";
import { arrangementToMidiBlob, parseMidiFile } from "./midi.js";
import { Player } from "./player.js";
import { NoteEditor } from "./editor.js";
import { drawChordStrip, highlightChord, barIndexAt } from "./views.js";

const MAX_TAKE_SECONDS = 90;
const MAX_UPLOAD_SECONDS = 300;

const $ = (id) => document.getElementById(id);

const state = {
  recorder: new Recorder(),
  recording: false,
  starting: false, // guards the async gap while the mic prompt is open
  vocalBuffer: null,
  analysis: null,
  arrangement: null,
  backingBuffer: null,
  modelCache: null, // per-take Basic Pitch output; cleared when the audio changes
  layerBuffers: null, // lazily rendered audition audio: { harmonics, prob }
  voiceTimbre: null, // the take's measured harmonic recipe (voice-timbre synth)
  mode: "full", // "full" = vocal + backing, "melody" = vocal + melody preview
  meterRaf: 0,
  playRaf: 0,
};

const player = new Player();
const editor = new NoteEditor($("wave"), { onChange: onNotesEdited });

// Pre-warm the transcriber (library + model download) while the user is
// still reading the page / recording, so the first analysis starts hot.
let modelReady = false;
warmup().then(() => { modelReady = true; });

// ---------- Recording ----------

$("recordBtn").addEventListener("click", async () => {
  if (!state.recording) {
    // The permission prompt makes start() slow the first time — on a public
    // site that is every visitor. Without this guard a second click during
    // the prompt starts a second capture and orphans the first one's stream,
    // leaving the mic live after Stop.
    if (state.starting) return;
    state.starting = true;
    try {
      await state.recorder.start();
    } catch (err) {
      showError("recError", `Microphone access failed: ${err.message}`);
      return;
    } finally {
      state.starting = false;
    }
    hideError("recError");
    state.recording = true;
    $("recordBtn").textContent = "■ Stop";
    $("recordBtn").classList.add("recording");
    drawMeter();
  } else {
    await finishRecording();
  }
});

async function finishRecording() {
  state.recording = false;
  cancelAnimationFrame(state.meterRaf);
  $("recordBtn").textContent = "● Record";
  $("recordBtn").classList.remove("recording");
  $("recordBtn").disabled = true;

  try {
    state.vocalBuffer = await state.recorder.stop();
    state.modelCache = null; // new audio: previous model output is stale
  } catch (err) {
    showError("recError", `Could not decode the recording: ${err.message}`);
    $("recordBtn").disabled = false;
    return;
  }
  $("recordBtn").disabled = false;

  if (state.vocalBuffer.duration < 3) {
    showError("recError", "That take was under 3 seconds — sing a bit longer so there's a melody to analyze.");
    return;
  }
  await runAnalysis();
}

function drawMeter() {
  const canvas = $("meter");
  const c = canvas.getContext("2d");
  const tickMeter = () => {
    if (!state.recording) {
      c.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const elapsed = state.recorder.elapsed();
    $("recTimer").textContent = formatTime(elapsed);
    if (elapsed >= MAX_TAKE_SECONDS) {
      finishRecording();
      return;
    }
    const level = state.recorder.level();
    c.clearRect(0, 0, canvas.width, canvas.height);
    c.fillStyle = level > 0.85 ? "#ff5d73" : "#5dd6ff";
    c.fillRect(0, 12, canvas.width * level, 24);
    state.meterRaf = requestAnimationFrame(tickMeter);
  };
  tickMeter();
}

// ---------- Upload ----------

$("uploadBtn").addEventListener("click", () => $("uploadInput").click());

$("uploadInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = ""; // allow re-selecting the same file later
  if (!file) return;
  hideError("recError");
  $("uploadName").textContent = `Decoding ${file.name}…`;

  const arrayBuffer = await file.arrayBuffer();
  const head = new Uint8Array(arrayBuffer.slice(0, 4));
  if (head[0] === 0x4d && head[1] === 0x54 && head[2] === 0x68 && head[3] === 0x64) {
    await handleMidiUpload(file, arrayBuffer); // "MThd": a MIDI file
    return;
  }

  let buf;
  try {
    const ctx = new AudioContext();
    buf = await ctx.decodeAudioData(arrayBuffer);
    await ctx.close();
  } catch (err) {
    $("uploadName").textContent = "";
    showError("recError", `Couldn't decode "${file.name}" — is it a valid audio file? (${err.message})`);
    return;
  }

  if (buf.duration < 3) {
    $("uploadName").textContent = "";
    showError("recError", "That file is under 3 seconds — use a longer take so there's a melody to analyze.");
    return;
  }
  if (buf.duration > MAX_UPLOAD_SECONDS) {
    buf = trimBuffer(buf, MAX_UPLOAD_SECONDS);
    $("uploadName").textContent = `${file.name} (analyzing the first ${MAX_UPLOAD_SECONDS / 60} minutes)`;
  } else {
    $("uploadName").textContent = file.name;
  }

  state.vocalBuffer = buf;
  state.modelCache = null; // new audio: previous model output is stale
  await runAnalysis();
});

/**
 * MIDI input skips transcription entirely: the file's notes and tempo map
 * are exact. A synthesized melody preview stands in for the vocal track so
 * playback, mixing, and export all work unchanged.
 */
async function handleMidiUpload(file, arrayBuffer) {
  try {
    const parsed = parseMidiFile(arrayBuffer);
    state.analysis = {
      notes: parsed.notes,
      key: detectKey(parsed.notes),
      tempo: parsed.tempo,
      beats: parsed.beats,
      tuningCents: 0,
      engine: "midi",
      firstOnset: parsed.notes[0].start,
      duration: parsed.duration,
    };
    state.vocalBuffer = await renderMelodyPreview(parsed.notes, 44100);
    $("uploadName").textContent = `${file.name} (${parsed.notes.length} notes)`;
  } catch (err) {
    $("uploadName").textContent = "";
    showError("recError", `Couldn't read "${file.name}": ${err.message}`);
    return;
  }
  $("analysis-panel").hidden = false;
  $("analysis-progress").hidden = true;
  $("analysis-results").hidden = false;
  $("player-panel").hidden = true;
  hideError("analysisError");
  fillAnalysisChips();
  $("analysis-panel").scrollIntoView({ behavior: "smooth" });
}

function trimBuffer(buf, seconds) {
  const length = Math.min(buf.length, Math.ceil(seconds * buf.sampleRate));
  const out = new AudioBuffer({
    length,
    sampleRate: buf.sampleRate,
    numberOfChannels: buf.numberOfChannels,
  });
  for (let c = 0; c < buf.numberOfChannels; c++) {
    out.copyToChannel(buf.getChannelData(c).subarray(0, length), c);
  }
  return out;
}

// ---------- Analysis ----------

function analysisSettings() {
  return {
    sensitivity: parseInt($("sensitivity").value, 10) / 100,
    snapOnsets: $("snapOnsets").checked,
    followNotes: $("followNotes").checked,
    autoTune: parseInt($("autoTune").value, 10) / 100,
    syllableLock: $("syllableLock").checked,
    minNote: parseFloat($("minNote").value),
  };
}

async function runAnalysis() {
  $("analysis-panel").hidden = false;
  $("analysis-progress").hidden = false;
  $("analysis-results").hidden = true;
  $("player-panel").hidden = true;
  hideError("analysisError");
  $("analysisBar").style.width = "0%";
  // Honest phase text: the first ever analysis may be waiting on the model
  // download, which used to look like a freeze.
  $("analysisPhase").textContent = state.modelCache?.frames
    ? "Re-analyzing with the new settings…"
    : modelReady
      ? "Listening to your performance…"
      : "Downloading the note-detection model (first run, a few MB)…";
  $("analysis-panel").scrollIntoView({ behavior: "smooth" });

  try {
    // The model cache makes settings changes near-instant: inference runs
    // once per take, and re-analyses only redo the fast post-processing.
    state.analysis = await analyzeVocal(state.vocalBuffer, (p) => {
      if (p > 0) $("analysisPhase").textContent = "Listening to your performance…";
      $("analysisBar").style.width = `${Math.round(p * 100)}%`;
    }, analysisSettings(), state.modelCache);
    state.modelCache = state.analysis.modelCache;
    state.layerBuffers = null; // notes changed: audition layers are stale
    state.voiceTimbre = null;
  } catch (err) {
    $("analysis-progress").hidden = true;
    showError("analysisError", err.message);
    return;
  }

  $("analysis-progress").hidden = true;
  $("analysis-results").hidden = false;
  fillAnalysisChips();
}

const ENGINE_LABELS = {
  "basic-pitch": "Basic Pitch (neural)",
  yin: "YIN (built-in)",
  midi: "MIDI file",
};

function fillAnalysisChips() {
  const cents = state.analysis.tuningCents;
  $("keyLabel").textContent =
    keyName(state.analysis.key) + (cents ? ` (${cents > 0 ? "+" : ""}${cents}¢)` : "");
  $("noteCount").textContent = String(state.analysis.notes.length);
  const engineLabel = ENGINE_LABELS[state.analysis.engine] ?? state.analysis.engine;
  const backend = state.analysis.backend;
  $("engineLabel").textContent =
    engineLabel + (backend ? ` · ${backend === "webgl" ? "GPU" : "CPU"}` : "");
  $("tempoInput").value = String(state.analysis.tempo);
  const lock = state.analysis.lockInfo;
  $("lockStatus").textContent = !lock || !lock.enabled
    ? ""
    : lock.engaged
      ? `✓${lock.dropped + lock.merged > 0 ? ` (−${lock.dropped + lock.merged})` : ""}`
      : "— varied";
  $("lockStatus").title = lock && lock.reason ? lock.reason : "";
}

// ---------- Generation ----------

// Re-analyze the same take when a transcription setting changes
// (not applicable to MIDI input, whose notes are exact).
function reanalyzeIfPossible() {
  if (state.vocalBuffer && state.analysis && state.analysis.engine !== "midi" && !state.recording) {
    runAnalysis();
  }
}
for (const id of ["sensitivity", "snapOnsets", "followNotes", "autoTune", "syllableLock", "minNote"]) {
  $(id).addEventListener("change", reanalyzeIfPossible);
}

function currentBeats() {
  const tempo = clamp(parseInt($("tempoInput").value, 10) || state.analysis.tempo, 50, 200);
  $("tempoInput").value = String(tempo);
  // Tracked beats follow the singer's drift; a manual tempo override
  // switches to a fixed grid at that BPM.
  return tempo === state.analysis.tempo
    ? state.analysis.beats
    : uniformBeats(state.analysis.firstOnset, state.analysis.duration, tempo);
}

// Melody-only mode skips accompaniment: it plays the detected notes (soft
// synth lead) against the vocal for comparison and exports just the melody
// MIDI. The arrangement is still computed for the MIDI tempo map and chord
// markers — only the audio backing differs.
const GEN = {
  full: {
    button: "generateBtn",
    statusWorking: "Arranging and rendering…",
    statusDone: "Done — scroll down to play.",
    makeBacking: (arr, samples, timbre) => renderBacking(arr, state.vocalBuffer.sampleRate, samples, timbre),
  },
  melody: {
    button: "melodyOnlyBtn",
    statusWorking: "Rendering melody preview…",
    statusDone: "Done — compare below; export the melody as MIDI.",
    makeBacking: () => renderMelodyPreview(state.analysis.notes, state.vocalBuffer.sampleRate),
  },
};

async function generate(mode) {
  const cfg = GEN[mode];
  const btn = $(cfg.button);
  btn.disabled = true;
  $("generateStatus").textContent = cfg.statusWorking;
  try {
    // Per-role instrument config: each role can carry its own voice.
    const voices = {
      chords: $("instrumentSelect").value,
      harmony: $("harmonySelect").value,
      bass: $("bassSelect").value,
      melody: $("melodySelect").value,
    };
    // Recorded instrument samples, CDN-loaded on first use. Failure never
    // blocks generation — the synth voices are the fallback, like YIN is
    // for the transcriber.
    let samples = null;
    if (mode === "full" && $("sampledSound").checked) {
      try {
        $("generateStatus").textContent = "Downloading instrument samples…";
        samples = await loadVoiceSamples(voices);
      } catch (err) {
        console.warn("Sampled instruments unavailable, using synth:", err);
      }
      $("generateStatus").textContent = cfg.statusWorking;
    }
    // The voice-timbre synth needs the take's measured harmonic recipe;
    // computed once per take (cleared with the layer buffers).
    let timbre = null;
    const resolveRole = (r) => (r === "follow" ? voices.chords : r);
    if (mode === "full" &&
        [voices.chords, resolveRole(voices.harmony), resolveRole(voices.melody)].includes("voice")) {
      state.voiceTimbre ??= voiceTimbre(
        resample(toMono(state.vocalBuffer), state.vocalBuffer.sampleRate, ANALYSIS_SR),
        state.analysis.notes,
      );
      timbre = state.voiceTimbre;
    }
    const arrangement = buildArrangement(
      { ...state.analysis, beats: currentBeats() },
      $("styleSelect").value,
      voices,
      { flourish: parseInt($("flourish").value, 10) / 100 },
    );
    state.arrangement = arrangement;
    state.backingBuffer = await cfg.makeBacking(arrangement, samples, timbre);
    state.mode = mode;
    showPlayer(arrangement);
  } catch (err) {
    $("generateStatus").textContent = "";
    showError("analysisError", `Generation failed: ${err.message}`);
    return;
  } finally {
    btn.disabled = false;
  }
  $("generateStatus").textContent = cfg.statusDone;
}

$("generateBtn").addEventListener("click", () => generate("full"));
$("melodyOnlyBtn").addEventListener("click", () => generate("melody"));
// "Apply edits" rebuilds the current mode's backing from the edited notes.
$("regenBtn").addEventListener("click", () => generate(state.mode));

// ---------- Note editing ----------

$("deleteNoteBtn").addEventListener("click", () => editor.deleteSelected());
$("mergeNotesBtn").addEventListener("click", () => editor.mergeSelected());
$("specToggle").addEventListener("change", () => editor.setSpectrogram($("specToggle").checked));

// Called by the editor after any selection or edit. The editor replaces its
// notes array on structural edits (delete/merge/sort), so adopt it here; key
// and note count are re-derived live, and the backing is marked stale until
// "Apply edits" regenerates it.
function onNotesEdited(e) {
  if (e.notes && state.analysis) state.analysis.notes = e.notes;
  const notes = state.analysis?.notes ?? [];
  $("deleteNoteBtn").disabled = !e.selCount || notes.length - e.selCount < 1;
  $("mergeNotesBtn").disabled = (e.selCount ?? 0) < 2;
  if (e.type !== "change") return;

  // Re-derive confidence shading: a note dragged somewhere the model heard
  // nothing should fade out. Needs one extra redraw (the editor drew first).
  if (noteConfidence(state.modelCache, notes)) editor.redraw();

  state.analysis.key = detectKey(notes);
  state.analysis.firstOnset = notes[0]?.start ?? 0;
  // Edited notes change the harmonic stack and the measured voice timbre;
  // the prob layers are audio-derived from the model and unaffected.
  if (state.layerBuffers) state.layerBuffers.harmonics = null;
  state.voiceTimbre = null;
  const cents = state.analysis.tuningCents;
  $("keyLabel").textContent =
    keyName(state.analysis.key) + (cents ? ` (${cents > 0 ? "+" : ""}${cents}¢)` : "");
  $("noteCount").textContent = String(notes.length);
  $("regenBtn").hidden = false;
  $("generateStatus").textContent = "Notes edited — Apply edits to update the backing.";
}

// ---------- Player panel ----------

function showPlayer(arrangement) {
  const melodyMode = state.mode === "melody";
  $("player-panel").hidden = false;
  $("chordStrip").style.display = melodyMode ? "none" : "";
  $("backingLabel").textContent = melodyMode ? "Melody" : "Backing";
  // Backing-dependent exports make no sense in melody-only mode.
  $("downloadBackingBtn").hidden = melodyMode;
  $("downloadMidiBtn").hidden = melodyMode;
  $("downloadMidiBackingBtn").hidden = melodyMode;
  // Probability layers need the model's posteriorgram (neural takes only).
  const noProb = !state.modelCache?.frames?.length;
  $("downloadMidiLayersBtn").disabled = noProb;
  $("probGain").disabled = noProb;
  noteConfidence(state.modelCache, state.analysis.notes);
  editor.load(state.vocalBuffer, state.analysis.notes);
  $("regenBtn").hidden = true;
  $("deleteNoteBtn").disabled = true;
  $("mergeNotesBtn").disabled = true;
  if (!melodyMode) drawChordStrip($("chordStrip"), arrangement.chordSymbols);
  player.stop();
  resetPlayButton();
  $("player-panel").scrollIntoView({ behavior: "smooth" });
}

/**
 * Renders the audition audio for a layer track on first use; cached per
 * analysis. Returns null when the layer has no data (e.g. probability
 * layers without a neural take).
 */
async function ensureLayerBuffer(name) {
  state.layerBuffers ??= {};
  if (state.layerBuffers[name]) return state.layerBuffers[name];
  let notes = null;
  if (name === "harmonics") {
    const x = resample(toMono(state.vocalBuffer), state.vocalBuffer.sampleRate, ANALYSIS_SR);
    notes = harmonicStack(x, state.analysis.notes);
  } else {
    notes = extractProbabilityNotes(state.modelCache);
  }
  if (!notes?.length) return null;
  state.layerBuffers[name] = await renderLayers(notes, state.vocalBuffer.sampleRate);
  return state.layerBuffers[name];
}

const LAYER_SLIDERS = [["harmonics", "harmonicsGain"], ["prob", "probGain"]];

$("playBtn").addEventListener("click", async () => {
  if (player.playing) {
    stopPlayback();
    return;
  }
  const tracks = {
    vocal: { buffer: state.vocalBuffer, gain: parseFloat($("vocalGain").value) },
    backing: { buffer: state.backingBuffer, gain: parseFloat($("backingGain").value) },
  };
  for (const [name, sliderId] of LAYER_SLIDERS) {
    const gain = parseFloat($(sliderId).value);
    if (gain > 0) {
      const buffer = await ensureLayerBuffer(name);
      if (buffer) tracks[name] = { buffer, gain };
    }
  }
  await player.play(tracks, {
    onEnded: () => {
      cancelAnimationFrame(state.playRaf);
      resetPlayButton();
    },
  });
  $("playBtn").textContent = "■ Stop";
  let lastBar = -1;
  const tickPlay = () => {
    const pos = player.position();
    $("playBar").style.width = `${Math.min(100, (pos / player.duration) * 100)}%`;
    // Follow the music through the chord strip.
    if (state.mode === "full" && state.arrangement) {
      const bar = barIndexAt(state.arrangement.bars, pos);
      if (bar !== lastBar) {
        highlightChord($("chordStrip"), bar);
        lastBar = bar;
      }
    }
    if (player.playing) state.playRaf = requestAnimationFrame(tickPlay);
  };
  tickPlay();
});

function stopPlayback() {
  player.stop();
  cancelAnimationFrame(state.playRaf);
  resetPlayButton();
}

function resetPlayButton() {
  $("playBtn").textContent = "▶ Play";
  $("playBar").style.width = "0%";
  highlightChord($("chordStrip"), -1);
}

$("vocalGain").addEventListener("input", () => player.setGain("vocal", parseFloat($("vocalGain").value)));
$("backingGain").addEventListener("input", () => player.setGain("backing", parseFloat($("backingGain").value)));

// Layer sliders: live gain, and if raised mid-playback before the track has
// ever been rendered, render it and join at the current position.
for (const [name, sliderId] of LAYER_SLIDERS) {
  $(sliderId).addEventListener("input", async () => {
    const v = parseFloat($(sliderId).value);
    player.setGain(name, v);
    if (v > 0 && player.playing && !player.hasTrack(name)) {
      const buffer = await ensureLayerBuffer(name);
      if (buffer && player.playing && !player.hasTrack(name)) player.addTrack(name, buffer, v);
    }
  });
}

// ---------- Export ----------

$("downloadMixBtn").addEventListener("click", () => {
  const blob = mixToWavBlob(
    state.vocalBuffer,
    state.backingBuffer,
    parseFloat($("vocalGain").value),
    parseFloat($("backingGain").value),
  );
  triggerDownload(blob, state.mode === "melody" ? "backtrack-melody-compare.wav" : "backtrack-mix.wav");
});

$("downloadBackingBtn").addEventListener("click", () => {
  const blob = mixToWavBlob(null, state.backingBuffer, 0, 1);
  triggerDownload(blob, "backtrack-backing.wav");
});

const midiExportOptions = () => ({ legato: $("legatoMidi").checked });
const MIDI_EXPORTS = [
  ["downloadMidiBtn", "full", "backtrack.mid"],
  ["downloadMidiBackingBtn", "backing", "backtrack-backing.mid"],
  ["downloadMidiMelodyBtn", "melody", "backtrack-melody.mid"],
];
for (const [id, parts, filename] of MIDI_EXPORTS) {
  $(id).addEventListener("click", () => {
    const blob = arrangementToMidiBlob(state.analysis, state.arrangement, parts, midiExportOptions());
    triggerDownload(blob, filename);
  });
}

$("downloadMidiHarmonicsBtn").addEventListener("click", () => {
  // Measure the singer's real overtones straight from the audio — works for
  // any take (no model needed), including the YIN fallback.
  const x = resample(toMono(state.vocalBuffer), state.vocalBuffer.sampleRate, ANALYSIS_SR);
  const layerNotes = harmonicStack(x, state.analysis.notes);
  if (!layerNotes.length) {
    showError("analysisError", "No harmonics could be measured from this take.");
    return;
  }
  const blob = arrangementToMidiBlob(state.analysis, state.arrangement, "layers", {
    ...midiExportOptions(),
    layerNotes,
    layersName: "Harmonics",
  });
  triggerDownload(blob, "backtrack-harmonics.mid");
});

$("downloadMidiLayersBtn").addEventListener("click", () => {
  const layerNotes = extractProbabilityNotes(state.modelCache);
  if (!layerNotes?.length) {
    showError("analysisError", "No probability data for this take — layers need a neural-analyzed recording (not MIDI input or the YIN fallback).");
    return;
  }
  const blob = arrangementToMidiBlob(state.analysis, state.arrangement, "layers", {
    ...midiExportOptions(),
    layerNotes,
  });
  triggerDownload(blob, "backtrack-layers.mid");
});

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ---------- Reset ----------

$("newTakeBtn").addEventListener("click", () => {
  stopPlayback();
  state.vocalBuffer = null;
  state.analysis = null;
  state.arrangement = null;
  state.backingBuffer = null;
  state.modelCache = null;
  state.layerBuffers = null;
  state.voiceTimbre = null;
  $("analysis-panel").hidden = true;
  $("player-panel").hidden = true;
  $("recTimer").textContent = "0:00";
  $("uploadName").textContent = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
});

// ---------- Helpers ----------

function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function showError(id, msg) {
  const el = $(id);
  el.textContent = msg;
  el.hidden = false;
}

function hideError(id) {
  $(id).hidden = true;
}

// Local-only handle for headless verification (no effect in production).
if (location.hostname === "localhost") window.__bt = { state, editor, generate, player };
