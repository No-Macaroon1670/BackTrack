# BackTrack

**Hum a melody, get a band.** Sing or hum into your mic and BackTrack
transcribes the notes, works out the key and tempo, fits a chord
progression, and plays an arrangement — chords, bass, drums, a harmony
voice — back underneath your vocal. Then export the mix as WAV or the whole
score as MIDI.

**▶ Try it: https://no-macaroon1670.github.io/BackTrack/**

Everything runs in your browser: your audio is never uploaded, and there is
no account, no server, and no build step. The only things fetched from the
network are the note-detection model and the instrument samples, pulled from
a public CDN the first time you need them and cached by the browser
afterwards.

Works in Chrome, Edge, and Firefox on the desktop. A first analysis
downloads a few MB of model weights, so give it a moment on a cold start.

## Run it locally

Microphone access needs a secure context, so serve the folder rather than
opening `index.html` from disk. Any static server works:

```bash
python3 -m http.server 4200
```

Then open <http://localhost:4200/>. On Windows there is a dependency-free
PowerShell server in the repo:

```
powershell -ExecutionPolicy Bypass -File serve.ps1
```

(If the port is taken or excluded by Windows — check with
`netsh int ipv4 show excludedportrange protocol=tcp` — pass `-Port <n>`.)
`serve.ps1`, `eval.ps1`, and `tools/*.ps1` are Windows conveniences only;
nothing in the app itself depends on them.

## How it works

1. **Record** (`js/recorder.js`) — captures the mic via `MediaRecorder` with
   voice processing (echo cancellation, noise suppression, AGC) disabled,
   then decodes the take into a raw `AudioBuffer`. Uploads accept audio
   files (MP3/WAV/M4A/OGG/FLAC) and **Standard MIDI Files**: MIDI input
   skips transcription entirely — non-drum tracks are pooled and reduced to
   the top melodic line, the file's tempo map becomes the beat grid, and a
   synthesized triangle-lead preview stands in for the vocal so playback,
   mixing, and export work unchanged.
2. **Analyze** (`js/analyze.js`, `js/transcribe.js`):
   - *Adaptive pitch sensitivity*: low voices have weak onset salience and
     the fixed detection threshold under-detects them. The pitch region is
     read straight from the cached posteriorgram (activation-weighted mean
     MIDI) and, for low-centroid takes only, the note-extraction threshold is
     lowered — no re-inference. Measured on Vocadito: low-voice clips gain
     ~0.05–0.10 COnP F (recall 0.29→0.46 on the worst), high voices unchanged.
   - *Note transcription*: Spotify's **Basic Pitch** neural model
     (TensorFlow.js build, downloaded from a CDN on first use, inference
     fully in-browser) detects notes and onsets, constrained to the vocal
     range and post-processed to a monophonic line. A note-hygiene pass
     then removes the model's spurious output: relative amplitude gating
     (breath blips), octave/twelfth ghost suppression, scoop-grace-note
     absorption, weak same-pitch continuation merging (genuine re-struck
     notes — zero gap, similar amplitude — are preserved), and
     melodic-outlier rejection. Every surviving note is cross-validated
     against the raw audio with YIN probes: notes with no periodicity
     (consonants, breaths) are dropped, octave confusions are corrected,
     and exact pitch is recovered (Basic Pitch quantizes to 1/3-semitone
     bins, too coarse for tuning). A **sensitivity slider** maps to the
     model's onset/frame thresholds and the amplitude gate — set it low and
     hum the main notes louder to keep only confident notes. A
     **peak-confidence gate** drops notes whose frame activation never
     spikes (threshold flicker), and a tempo-aware **"Min note" setting**
     (32nd/16th/8th) absorbs or removes anything shorter than the singer's
     declared smallest note value. The model's frame output is **cached per
     take**, so changing any analysis setting re-extracts notes in
     milliseconds instead of re-running inference. Detected notes are drawn over the waveform
     in the player so you can see what the analyzer heard, and a **"Melody
     only" mode** skips accompaniment entirely: it plays a soft synth lead
     of the detected notes against your vocal for comparison and exports
     the melody MIDI. If the model can't load (offline), the analyzer falls
     back to the built-in YIN tracker:
     cumulative mean normalized difference (threshold 0.15, parabolic
     interpolation), 64 ms frames, 16 ms hop, median filtering, and
     segmentation that splits on jumps greater than 0.7 semitones. The UI
     shows which engine was used ("Ears" chip).
   - *Tuning offset*: duration-weighted circular mean of each note's
     deviation from equal temperament; pitches are rounded relative to it
     and the backing is detuned by it, so the band plays in the singer's
     tuning rather than A=440.
   - *Key detection*: duration-weighted pitch-class histogram correlated
     against Krumhansl-Kessler major/minor profiles (24 candidates).
   - *Beat tracking*: onset envelope (rectified RMS derivative reinforced by
     note starts), global period by autocorrelation with a prior near
     100 BPM, then Ellis-style dynamic programming finds a beat sequence
     that follows the singer's tempo drift. Overriding the tempo in the UI
     switches to a fixed grid at that BPM.
   - *Onset snapping* (toggle): note starts snap to the beginning of the
     nearest significant energy rise in a fine 6 ms envelope — energy moves
     at the moment of articulation, before pitch stabilizes. Legato
     transitions (no rise) are left untouched.
   - *Auto-tune* is an intensity slider (0 = off): it scales how long a
     note can be and still count as an absorbable ornament (~0.06-0.3 s),
     how long an out-of-key note can be and still get scale-snapped
     (~0.16-0.8 s), and how wide a gap same-pitch fragments rejoin across.
     50 is the balanced default behavior; 100 simplifies aggressively.
   - *Syllable lock* (toggle): if every note is articulated with the same
     stop-consonant syllable ("Doo"), its acoustic signature — closure dip
     before the onset, fast release burst, consistent vowel timbre — is
     learned from the take itself (median ± MAD over all notes) and used
     to reject sounds that don't begin like a note and to settle
     split-vs-restrike questions by checking for a closure dip at the
     boundary. Engages only when the take's onsets measure consistent;
     stands down automatically on varied (lyric-like) articulation.
   - *Note-anchored grid* ("Follow my timing" toggle, default on): instead
     of trusting the energy-tracked grid alone, each inter-onset interval
     is classified as a musical duration (with a prior for metrically
     simple readings — a slightly-early entry after a half note still
     starts the next bar) and the grid re-anchors its time at every note,
     so a wandering internal tempo never accumulates error. Local tempo
     adapts smoothly, clamped per step.
3. **Arrange** (`js/arrange.js`) — 4/4 bars are anchored to the tracked
   beats, with the downbeat phase chosen so note onsets land on beats 1
   and 3 (pickups don't shift the grid). Chords are fitted at half-bar
   granularity by Viterbi decoding over the six diatonic triads:
   melody pitch classes are weighted by duration and metric position
   (strong-beat notes count double, likely passing tones are discounted),
   with functional-harmony transition bonuses (ii→V, V→I, …), a per-change
   penalty so chords hold unless the melody insists, an extra penalty for
   mid-bar changes, and cadence/tonic-start bonuses. The progression is
   expanded into pad/bass/drum events using per-style patterns (Pop,
   Ballad, Lo-fi with swing and 7th chords, Folk with strums) and
   smooth-voice-leading chord voicings, all timed from the tracked beats.
   Instruments are assigned **per role**: Chords (keys / guitar / strings /
   your voice — this choice also sets the accompaniment pattern), Harmony
   (any of those, follow-chords, or off), Bass (acoustic / electric /
   synth / off), and Melody — an optional instrument doubling the sung
   line at the singer's exact timing. Each role loads its own soundfont;
   MIDI export gives harmony and melody-double their own tracks with
   matching programs. The **Flourish** slider ornaments the Melody
   instrument (never the vocal or the transcribed melody): scale-aware
   passing tones fill leaps, held notes get neighbor-and-return figures,
   busy bars are left alone, and the amount scales both frequency and
   elaborateness — seeded, so the same take and slider always render the
   same.
   A **harmony voice** (Harmony role, on by default) shadows the sung
   melody like a bandmate singing along: each sustained note gets a
   companion a third below (occasionally a sixth), chord tones first and
   scale tones as the fallback; notes that would sink below E2 harmonize a
   third above instead, and fast runs are left alone. It plays with the
   singer's exact timing — so it inherits all the phrasing and can never
   clash rhythmically — and exports as its own MIDI track ("Harmony
   voice") for DAW work.
   Chords pick up **color the melody earns**: if the singer leans on the
   9th over a chord it becomes an add9, avoiding the 3rd while leaning on
   the 4th makes it a sus4, a prominent major 7th over I or IV adds it —
   and V resolving to I always gets its cadential 7th. At most one color
   per half-bar, chord choice is never affected, and the chord strip shows
   the resulting names (Badd9, B7, …).
   Chord evidence is anticipation-aware: a note sung up to an eighth ahead
   of a boundary and held across it votes for the *next* segment's chord,
   not the one it merely started in — so syncopated phrasing picks the
   chord the singer meant. Arpeggio figures yield to a busy voice (bars
   with more than one onset per beat hold the chord instead), and the bass
   occasionally walks to the octave instead of the fifth.
   Where a chord *changes*, the sustained chord voice snaps its entry to the
   singer's nearest note onset within ±120 ms — a real accompanist moves
   with the singer's phrasing — while bass, drums, strums and arpeggios
   hold the grid. A light humanization pass (a few milliseconds of timing
   looseness and small velocity variation, seeded so renders are
   reproducible) keeps the comping from sounding sample-accurate; the
   ending stays exact. A dynamics layer then makes the band *respond to
   the performance*:
   every bar's velocities follow the vocal's own energy (amplitude-weighted
   coverage, smoothed and normalized), drum patterns thin out when the
   voice pulls back, drums sit out the first bar as an intro, sung phrase
   boundaries (rests ≥ ~1.3 beats) earn a drum fill into a crash on the
   next downbeat, and the take ends on a held chord with kick + crash
   rather than stopping dead. The dynamics flow into the MIDI export as
   velocities, so a DAW rendering breathes the same way.
4. **Render** (`js/render.js`) — events are scheduled into an
   `OfflineAudioContext`. The **"Your voice (choir)"** backing option is a
   voice-timbre synth: the take's average overtone recipe is measured from
   the audio (`notes.voiceTimbre`, the same per-note partial measurement as
   the MIDI-harmonics export, duration-weighted) and turned into a
   `PeriodicWave` oscillator — the chord pads and harmony voice become a
   choir with the singer's own spectrum, with choir attacks and
   delayed-onset vibrato. With **Sampled sound** on (the default),
   the chord and bass voices play real recorded notes: the MIT-licensed
   FluidR3 General-MIDI soundfont (via the midi-js-soundfonts project),
   fetched once from a CDN like the transcription model, decoded and
   cached per session (`js/sampler.js`), per-font peak-normalized, detuned
   to the singer's tuning, and driven by the same event/velocity pipeline
   as the synths. Unticked — or offline — the original synth voices render
   instead: detuned saw pads through a lowpass (keys), Karplus-Strong
   plucks (acoustic guitar), or bowed strings with slow attacks and
   delayed-onset vibrato (violin), plus triangle bass. Drums stay
   synthesized in both modes (kick/snare/hat/crash), through a send reverb
   with a generated impulse response and a master compressor. Playback
   mixes vocal and backing live with independent gains; export encodes a
   16-bit stereo WAV.
5. **Probability-layers MIDI** — instead of collapsing the model's output to
   one winning line, "MIDI prob layers" exports *every* pitch the
   posteriorgram considered plausible, with **velocity encoding its
   probability** (gamma-lifted, so the melody comes out loud, octave and
   harmonic candidates quiet, ambiguity as soft texture). Made for layering
   and mixing in a DAW; needs a neural-analyzed take (disabled for MIDI
   input / YIN fallback). Its sibling, **"MIDI harmonics"**, measures each
   melody note's *real* overtones from the audio (FFT at the note's stable
   centre, partial peaks within ±3% of k·f0) and exports the stack with
   velocity = measured partial strength relative to the note's loudest
   partial — the singer's own timbre as playable MIDI. Works for any take.
   Both layer types are auditionable in the player: **Harmonics** and
   **Prob layers** volume sliders (default 0) render the layer to soft
   sines on first use — with the same probability→gain curve the MIDI
   export uses for velocity — and can join live mid-playback.
6. **MIDI export** (`js/midi.js`) — writes a format-1 Standard MIDI File for
   notation/DAW software (MuseScore, Logic, Ableton, …): a conductor track
   with a per-beat tempo map (so the tracked tempo drift lands on proper
   bars), 4/4 time signature, key signature, and chord-name markers; then
   vocal melody, chords (piano or steel guitar per the chosen instrument),
   bass, and drums (channel 10). Notes are quantized to a 16th grid, with a
   pickup bar prepended when the melody starts before the first downbeat.

## Module layout

Vanilla ES modules, no build step. The analysis pipeline is split by concern
so each stage is testable in isolation:

- `js/dsp.js` — shared primitives: mono downmix, resampling, running
  statistics, and the YIN pitch estimator.
- `js/transcribe.js` — Basic Pitch model loading, GPU-inline vs. CPU-worker
  backend selection, and raw-output → note-list extraction.
- `js/transcribe.worker.js` — CPU inference off the main thread.
- `js/notes.js` — every note-list transform: model-output hygiene
  (`cleanNotes`), YIN cross-validation, held-note consolidation (rejoins
  same-pitch fragments when the audio shows continuous energy across the
  junction; a real re-articulation dips at the closure and survives),
  onset snapping, syllable lock, minimum-note filtering, and auto-tune
  simplification.
- `js/rhythm.js` — beat tracking, tempo folding, and the note-anchored grid.
- `js/key.js` — tuning-offset and Krumhansl-Schmuckler key detection.
- `js/analyze.js` — orchestrates the above into `analyzeVocal`, and hosts the
  YIN fallback engine.
- `js/arrange.js`, `js/render.js`, `js/midi.js`, `js/recorder.js`
  — arrangement, synthesis, MIDI I/O, and capture.
- `js/player.js` — DOM-free two-track playback engine (vocal + backing
  gains, transport position).
- `js/views.js` — player-panel rendering: waveform + note overlay, chord
  strip, playback bar highlighting, and the note↔pixel layout math.
- `js/editor.js` — interactive note editing over the waveform: click /
  shift-click / rubber-band marquee to select, drag to move notes in pitch
  (semitone-snapped) and time (free), drag a note's edge to resize,
  double-click to add, Delete removes the selection, arrows nudge pitch
  (↑↓) or timing (←→, shift = coarse), and ⧉ Merge fuses the selection into
  one note — the manual companion to the analyzer's consolidation pass.
  Bar brightness shows the analyzer's confidence in each note (the neural
  model's posteriorgram support for the note as drawn, recomputed after every
  edit — drag a note somewhere the model heard nothing and it fades). The
  key chip re-derives live and "Apply edits" rebuilds the backing from the
  corrected melody.
- `js/spectrogram.js` — a self-contained radix-2 FFT and STFT that renders a
  log-frequency (constant-Q-style) spectrogram, mapped through the same
  pitch layout as the note overlay so harmonic stacks align vertically with
  the detected notes. Toggleable underlay in the editor; an octave error
  shows as a note bar sitting on a gap with a bright line an octave below.
- `js/app.js` — UI wiring only: element ids, event listeners, and app state.
  Pre-warms the transcriber (library + model download) at page load so the
  first analysis starts hot, and shows honest phase text ("Downloading the
  note-detection model…") when it doesn't.

## Accuracy evaluation

`eval.html` (with `js/eval.js` + `js/metrics.js`) scores the transcription
against a real vocal dataset using mir_eval-faithful metrics — note-level
precision/recall/F (onset-only, onset+pitch, onset+pitch+offset, via maximum
bipartite matching) and frame-level raw pitch/chroma accuracy and voicing.
Datasets are large and licensed, so none ship here: download one and point
the file pickers at it. Supported annotations, matched to audio by filename:

- **Vocadito** (CC-BY, solo singing) — F0 CSV and note CSV. Best fit.
- **MIR-1K** — `.pv` frame pitch (free for research).
- **MIREX ADC2004 / MIREX05** — F0 `time,freq` text.

**Gap recovery.** The measured weakness was recall — on the hardest clips the
transcriber found barely a third of the reference notes, and the settings
sweep showed a globally higher sensitivity trades that recall away elsewhere.
So a second pass re-reads the *cached* posteriorgram only where nothing was
detected, takes the argmax pitch per frame, and keeps runs that hold a stable
pitch; every candidate then faces the same YIN cross-validation as the rest.
On Vocadito that is worth **+7.0 onset F, +9.0 recall, +3.6 COnP** — and
precision rises too (+2.3), so it is not the usual recall-for-precision
trade. Its threshold (0.18) is a measured optimum with accuracy falling off
on both sides; counter-intuitively a *stricter* threshold recovers *more*,
because chasing the argmax through low-confidence frames yields wobbling
pitch that fragments into unusable runs.

**What a good score actually is.** Vocadito ships two independent human
annotations, so the harness can score one annotator against the other and
ask what "perfect" looks like. The answer is **not** 1.0: two experts agree
at onset F 0.82, COnP 0.74, COnPOff 0.64, and disagree about how many notes
a clip even contains by roughly 16%. Measured against that ceiling this
transcriber reaches **86% of human agreement on onsets, 72% on onset+pitch,
and 60% on offsets** — so onsets are close to saturated and offsets hold
most of the remaining headroom. Per-clip difficulty also correlates (r ≈
0.47) with how much the two humans disagree: the hard clips are hard for
everyone. Raw F-measures here should always be read against that ceiling
rather than against a perfect score.

Note-level F-measures are the fair score for a note transcriber; frame RPA
reads lower because note transcription intentionally omits glissandi a
continuous-F0 reference includes. The "Self-test metrics" button proves the
whole path (decode → analyze → score) on a synthetic clip, no dataset needed.

**Autorun.** Put the dataset under `backtrack/vocadito/` (served over
localhost), then:

```
powershell -ExecutionPolicy Bypass -File eval.ps1            # score the set
powershell -ExecutionPolicy Bypass -File eval.ps1 -Compare   # sweep configs
```

`eval.ps1` regenerates `vocadito/manifest.json` (via
`tools/build-eval-manifest.ps1`, so the browser can list the folder), ensures
the server is up, and opens `eval.html?run=vocadito`, which auto-loads and
scores every clip. **Compare configs** runs a set of analysis settings over
the dataset — reusing one model inference per clip, so a six-config sweep
costs barely more than a single run — and tabulates them so you can see which
settings move accuracy. "⬇ Results JSON" saves a baseline to diff across
builds.

## Ideas for v2

- Live accompaniment mode (AudioWorklet pitch tracking, look-ahead chords).
- Rests as phrase boundaries with cadence bonuses at phrase ends.
- Wider harmonic palette: bVII in major, secondary dominants, modal mixture.
- 3/4 and 6/8 meters (the grid is 4/4 only today).
- AI-generated stems layered over the algorithmic arrangement.

## Credits

BackTrack's own code is original, but it stands on two pieces of other
people's work, both fetched at runtime from a CDN:

- **[Spotify Basic Pitch](https://github.com/spotify/basic-pitch)** — the
  neural note-transcription model, via its TensorFlow.js build.
  Licensed Apache-2.0.
- **FluidR3_GM** — the General MIDI soundfont behind the sampled piano,
  guitar, strings, and bass, pre-rendered per note by
  [gleitz/midi-js-soundfonts](https://github.com/gleitz/midi-js-soundfonts).
  Licensed [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).

Evaluation uses the [Vocadito](https://zenodo.org/records/5578807) dataset
(CC BY 4.0), which is **not** redistributed here — download it separately if
you want to run the accuracy harness.

BackTrack itself is released under the [MIT License](LICENSE).
