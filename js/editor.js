// Interactive note editor over the waveform canvas. Turns the read-only
// detected-note overlay into an editing surface:
//   - click selects; shift-click toggles; dragging on empty space rubber-band
//     selects everything the marquee touches
//   - dragging a note moves the whole selection in time (free) and pitch
//     (snapped to semitones); dragging a note's edge resizes that note
//   - double-click empty space adds a note; Delete removes the selection;
//     arrows nudge pitch (up/down) or timing (left/right, shift = coarse)
//   - mergeSelected() fuses the selection into one note (duration-weighted
//     pitch), the manual companion to the analyzer's consolidation pass
// Mutates the notes array in place and reports changes via onChange so
// app.js can update the key chip and mark the backing stale.

import { drawWaveform, noteLayout } from "./views.js";
import { computeSpectrogram } from "./spectrogram.js";

const ADD_NOTE_DUR = 0.4; // seconds, for notes created by double-click
const EDGE_PX = 4; // horizontal tolerance for grabbing a note edge
const MIN_DUR = 0.05; // resizing can't make a note shorter than this
const NUDGE_SEC = 0.02; // arrow-key timing nudge (shift = 5x)

export class NoteEditor {
  constructor(canvas, { onChange } = {}) {
    this.canvas = canvas;
    this.onChange = onChange ?? (() => {});
    this.buffer = null;
    this.notes = [];
    this.duration = 0;
    this.range = null; // fixed pitch window { lo, hi } for stable dragging
    this.selection = new Set(); // indices into notes
    this.enabled = false;
    this.showSpec = true;
    this.specCanvas = null; // cached offscreen spectrogram (audio-derived; survives edits)

    this.drag = null; // { mode: "move"|"resize-l"|"resize-r"|"marquee", ... }
    this._bind();
  }

  /** Attach to a take: `notes` is mutated in place as the user edits. */
  load(buffer, notes) {
    this.buffer = buffer;
    this.notes = notes;
    this.duration = buffer.duration;
    this.selection.clear();

    // Fixed pitch range (detected span padded to at least an octave) so a
    // note doesn't visually rescale the others while you drag it.
    let lo = Infinity, hi = -Infinity;
    for (const n of notes) { lo = Math.min(lo, n.midi); hi = Math.max(hi, n.midi); }
    if (!Number.isFinite(lo)) { lo = 57; hi = 69; }
    lo -= 4; hi += 4;
    while (hi - lo < 12) { hi++; lo--; }
    this.range = { lo, hi };
    this.enabled = true;

    // Spectrogram is audio-derived, so compute once per take (not per edit),
    // aligned to the same pitch layout as the note overlay. Never fatal.
    this.specCanvas = null;
    try {
      const layout = noteLayout(this.canvas, this.duration, notes, this.range);
      this.specCanvas = computeSpectrogram(buffer, this.canvas.width, this.canvas.height, layout);
    } catch (err) {
      console.warn("Spectrogram unavailable:", err);
    }
    this.redraw();
  }

  setSpectrogram(on) {
    this.showSpec = on;
    this.redraw();
  }

  redraw() {
    if (!this.buffer) return;
    drawWaveform(this.canvas, this.buffer, this.notes, {
      selectedIndices: this.selection,
      range: this.range,
      background: this.showSpec ? this.specCanvas : null,
      marquee: this.drag?.mode === "marquee" ? this.drag.rect : null,
    });
  }

  /** Replaces the selection with a single index (or clears it with -1). */
  select(i) {
    this.selection.clear();
    if (i >= 0) this.selection.add(i);
    this.redraw();
    this._selected();
  }

  deleteSelected() {
    if (!this.selection.size || this.notes.length - this.selection.size < 1) return;
    this.notes = this.notes.filter((_, i) => !this.selection.has(i));
    this._commitNotes();
  }

  /** Fuses the selected notes into one spanning note. */
  mergeSelected() {
    if (this.selection.size < 2) return;
    const picked = [...this.selection].map((i) => this.notes[i]);
    let start = Infinity, end = -Infinity, wSum = 0, mSum = 0, amp = 0;
    for (const n of picked) {
      start = Math.min(start, n.start);
      end = Math.max(end, n.end);
      const w = Math.max(0.01, n.end - n.start);
      wSum += w;
      mSum += (n.midiFloat ?? n.midi) * w;
      amp = Math.max(amp, n.amp ?? 0);
    }
    const midi = Math.round(mSum / wSum);
    const merged = { start, end, dur: end - start, midi, midiFloat: midi, amp };
    this.notes = this.notes.filter((_, i) => !this.selection.has(i));
    this.notes.push(merged);
    this._commitNotes(merged);
  }

  // ---- internals ----

  /** Re-sorts after a structural edit, restores selection, fires change. */
  _commitNotes(...keepSelected) {
    this.notes.sort((a, b) => a.start - b.start);
    this.selection = new Set(keepSelected.map((n) => this.notes.indexOf(n)).filter((i) => i >= 0));
    this.redraw();
    this._changed();
  }

  _layout() {
    return noteLayout(this.canvas, this.duration, this.notes, this.range);
  }

  _canvasXY(e) {
    const r = this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * (this.canvas.width / r.width),
      y: (e.clientY - r.top) * (this.canvas.height / r.height),
    };
  }

  /** Index of the note under (px, py), or -1. Nearest in pitch on ties. */
  _hit(px, py) {
    const L = this._layout();
    let best = -1, bestDy = 9; // px vertical tolerance (bars are ~3px tall)
    for (let i = 0; i < this.notes.length; i++) {
      const n = this.notes[i];
      if (px < L.toX(n.start) - 3 || px > L.toX(n.end) + 3) continue;
      const dy = Math.abs(py - L.toY(n.midi));
      if (dy < bestDy) { bestDy = dy; best = i; }
    }
    return best;
  }

  /** "resize-l" | "resize-r" | null for a hit note under px. */
  _edge(i, px) {
    if (i < 0) return null;
    const L = this._layout();
    const n = this.notes[i];
    const x0 = L.toX(n.start), x1 = L.toX(n.end);
    if (x1 - x0 < 3 * EDGE_PX) return null; // too narrow: whole bar moves
    if (Math.abs(px - x0) <= EDGE_PX) return "resize-l";
    if (Math.abs(px - x1) <= EDGE_PX) return "resize-r";
    return null;
  }

  _nudgePitch(delta) {
    let moved = false;
    for (const i of this.selection) {
      const n = this.notes[i];
      const midi = Math.max(this.range.lo, Math.min(this.range.hi, n.midi + delta));
      if (midi !== n.midi) { n.midi = midi; n.midiFloat = midi; moved = true; }
    }
    if (moved) { this.redraw(); this._changed(); }
  }

  _nudgeTime(dt) {
    const picked = [...this.selection].map((i) => this.notes[i]);
    if (!picked.length) return;
    let lo = Infinity, hi = -Infinity;
    for (const n of picked) { lo = Math.min(lo, n.start); hi = Math.max(hi, n.end); }
    const d = Math.max(-lo, Math.min(this.duration - hi, dt));
    if (!d) return;
    for (const n of picked) { n.start += d; n.end += d; }
    this._commitNotes(...picked);
  }

  _addAt(px, py) {
    const L = this._layout();
    const start = Math.max(0, Math.min(this.duration - 0.1, L.fromX(px)));
    const midi = Math.round(L.fromY(py));
    const note = { start, end: Math.min(this.duration, start + ADD_NOTE_DUR), midi, midiFloat: midi, amp: 0.8 };
    note.dur = note.end - note.start;
    this.notes.push(note);
    this._commitNotes(note);
  }

  _changed() {
    this.onChange({ type: "change", selCount: this.selection.size, count: this.notes.length, notes: this.notes });
  }

  _selected() {
    this.onChange({ type: "select", selCount: this.selection.size, count: this.notes.length, notes: this.notes });
  }

  _startMove(x, y) {
    const picked = [...this.selection];
    this.drag = {
      mode: "move", x0: x, y0: y, moved: false,
      orig: picked.map((i) => ({ i, start: this.notes[i].start, end: this.notes[i].end, midi: this.notes[i].midi })),
    };
  }

  _bind() {
    this.canvas.addEventListener("mousedown", (e) => {
      if (!this.enabled) return;
      this.canvas.focus();
      e.preventDefault();
      const { x, y } = this._canvasXY(e);
      const hit = this._hit(x, y);

      if (hit < 0) {
        // Empty space: rubber-band select (shift keeps the current selection).
        this.drag = { mode: "marquee", x0: x, y0: y, rect: { x0: x, y0: y, x1: x, y1: y }, keep: e.shiftKey ? new Set(this.selection) : new Set(), moved: false };
        if (!e.shiftKey && this.selection.size) { this.selection.clear(); this.redraw(); this._selected(); }
        return;
      }

      if (e.shiftKey) {
        // Toggle membership; no drag from a shift-click.
        if (this.selection.has(hit)) this.selection.delete(hit);
        else this.selection.add(hit);
        this.redraw();
        this._selected();
        return;
      }

      const edge = this._edge(hit, x);
      if (edge) {
        this.selection = new Set([hit]);
        this.redraw();
        this._selected();
        this.drag = { mode: edge, i: hit, start: this.notes[hit].start, end: this.notes[hit].end, x0: x, moved: false };
        return;
      }

      if (!this.selection.has(hit)) {
        this.selection = new Set([hit]);
        this.redraw();
        this._selected();
      }
      this._startMove(x, y);
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.enabled) {
        return;
      }
      if (!this.drag) {
        // Hover feedback only when the pointer is over the canvas.
        if (e.target === this.canvas) {
          const { x, y } = this._canvasXY(e);
          const hit = this._hit(x, y);
          const edge = this._edge(hit, x);
          this.canvas.style.cursor = edge ? "ew-resize" : hit >= 0 ? "pointer" : "crosshair";
        }
        return;
      }
      const { x, y } = this._canvasXY(e);
      const L = this._layout();
      const d = this.drag;

      if (d.mode === "marquee") {
        d.rect = { x0: d.x0, y0: d.y0, x1: x, y1: y };
        d.moved = d.moved || Math.abs(x - d.x0) + Math.abs(y - d.y0) > 4;
        const [rx0, rx1] = [Math.min(d.x0, x), Math.max(d.x0, x)];
        const [ry0, ry1] = [Math.min(d.y0, y), Math.max(d.y0, y)];
        this.selection = new Set(d.keep);
        this.notes.forEach((n, i) => {
          const nx0 = L.toX(n.start), nx1 = L.toX(n.end), ny = L.toY(n.midi);
          if (nx1 >= rx0 && nx0 <= rx1 && ny >= ry0 - 3 && ny <= ry1 + 3) this.selection.add(i);
        });
        this.redraw();
        return;
      }

      if (d.mode === "resize-l" || d.mode === "resize-r") {
        const dt = L.fromX(x) - L.fromX(d.x0);
        const n = this.notes[d.i];
        if (d.mode === "resize-l") n.start = Math.max(0, Math.min(d.end - MIN_DUR, d.start + dt));
        else n.end = Math.min(this.duration, Math.max(d.start + MIN_DUR, d.end + dt));
        n.dur = n.end - n.start;
        d.moved = true;
        this.redraw();
        return;
      }

      // move: horizontal = time (free), vertical = pitch (semitone-snapped)
      const dtRaw = L.fromX(x) - L.fromX(d.x0);
      let lo = Infinity, hi = -Infinity;
      for (const o of d.orig) { lo = Math.min(lo, o.start); hi = Math.max(hi, o.end); }
      const dt = Math.max(-lo, Math.min(this.duration - hi, dtRaw));
      const perPx = (this.range.hi - this.range.lo) / (this.canvas.height - 12);
      const dSemis = Math.round((d.y0 - y) * perPx);
      for (const o of d.orig) {
        const n = this.notes[o.i];
        n.start = o.start + dt;
        n.end = o.end + dt;
        n.dur = n.end - n.start;
        const midi = Math.max(this.range.lo, Math.min(this.range.hi, o.midi + dSemis));
        n.midi = midi;
        n.midiFloat = midi;
      }
      if (dt !== 0 || dSemis !== 0) d.moved = true;
      this.redraw();
    });

    window.addEventListener("mouseup", () => {
      if (!this.drag) return;
      const d = this.drag;
      this.drag = null;
      if (d.mode === "marquee") {
        this.redraw(); // clears the marquee rect
        this._selected();
      } else if (d.moved) {
        if (d.mode === "move") {
          const picked = d.orig.map((o) => this.notes[o.i]);
          this._commitNotes(...picked); // re-sort: time order may have changed
        } else {
          this.redraw();
          this._changed();
        }
      }
    });

    this.canvas.addEventListener("dblclick", (e) => {
      if (!this.enabled) return;
      const { x, y } = this._canvasXY(e);
      if (this._hit(x, y) >= 0) return; // on a note: leave deletion to the button/key
      this._addAt(x, y);
    });

    this.canvas.addEventListener("keydown", (e) => {
      if (!this.enabled || !this.selection.size) return;
      const coarse = e.shiftKey ? 5 : 1;
      if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); this.deleteSelected(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); this._nudgePitch(1); }
      else if (e.key === "ArrowDown") { e.preventDefault(); this._nudgePitch(-1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); this._nudgeTime(-NUDGE_SEC * coarse); }
      else if (e.key === "ArrowRight") { e.preventDefault(); this._nudgeTime(NUDGE_SEC * coarse); }
    });
  }
}
