// Web Worker: runs Basic Pitch inference on TF.js's CPU backend off the main
// thread, so machines without a usable GPU stay responsive during analysis
// (the CPU backend is synchronous and would otherwise freeze the page for the
// whole run). transcribe.js only uses this worker when there's no hardware
// WebGL — with a real GPU it runs WebGL inference inline instead, which is
// faster. (WebGL inside a Worker via OffscreenCanvas was measured slower than
// CPU here, so the worker deliberately stays on CPU.) Falls back to inline
// inference in transcribe.js if this worker can't start.

const BP_VERSION = "1.0.1";
const LIB_URL = `https://esm.sh/@spotify/basic-pitch@${BP_VERSION}?bundle`;
const MODEL_URL = `https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@${BP_VERSION}/model/model.json`;

let libPromise = null;
const loadLib = () => (libPromise ??= import(/* webpackIgnore: true */ LIB_URL));

let bpInstance = null;

self.onmessage = async (e) => {
  if (!e.data || e.data.type !== "infer") return;
  try {
    const { BasicPitch } = await loadLib();
    const frames = [], onsets = [], contours = [];
    const bp = (bpInstance ??= new BasicPitch(MODEL_URL));
    await bp.evaluateModel(
      e.data.audio,
      (f, o, c) => { frames.push(...f); onsets.push(...o); contours.push(...c); },
      (p) => self.postMessage({ type: "progress", p }),
    );
    self.postMessage({ type: "done", frames, onsets, contours });
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};
