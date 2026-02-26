/**
 * GPU Profiler using EXT_disjoint_timer_query_webgl2
 *
 * Wraps WebGL2 timer queries to measure per-pass GPU execution time.
 * Queries are async — results are read back ~2 frames after submission.
 *
 * Usage:
 *   const profiler = new GPUProfiler(gl);
 *   profiler.beginPass("predictLocal_1");
 *   renderer.render(scene, camera);
 *   profiler.endPass();
 *   // ... later each frame:
 *   profiler.frameEnd();
 *   console.log(profiler.getSummary());
 */

export class GPUProfiler {
  /**
   * @param {WebGL2RenderingContext} gl
   * @param {number} [ringSize=120] — number of frames to keep in the ring buffer
   */
  constructor(gl, ringSize = 120) {
    this.gl = gl;
    this.ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    this.available = !!this.ext;
    this.enabled = this.available;
    this.ringSize = ringSize;

    // Pending queries: array of { label, query } submitted but not yet resolved
    this._pending = [];
    // Resolved per-frame pass times: ring buffer of Map<label, microseconds>
    this._frames = [];
    // Current frame's pass accumulator
    this._currentFrame = new Map();
    // Currently active pass label + query
    this._activeLabel = null;
    this._activeQuery = null;
    // Pass count per frame for current frame
    this._passCount = 0;
    // Total passes ring buffer
    this._passCountRing = [];

    // Pre-populate counters for known pass labels
    this._knownLabels = new Set();
  }

  /** Start timing a named render pass. */
  beginPass(label) {
    if (!this.enabled) return;
    const gl = this.gl;

    // Check for disjoint — GPU timer may have been reset
    if (gl.getParameter(this.ext.GPU_DISJOINT_EXT)) {
      // Discard all pending queries
      this._pending.forEach((p) => gl.deleteQuery(p.query));
      this._pending = [];
    }

    const query = gl.createQuery();
    gl.beginQuery(this.ext.TIME_ELAPSED_EXT, query);
    this._activeLabel = label;
    this._activeQuery = query;
    this._knownLabels.add(label);
  }

  /** End the current pass timer. */
  endPass() {
    if (!this.enabled || !this._activeQuery) return;
    this.gl.endQuery(this.ext.TIME_ELAPSED_EXT);
    this._pending.push({
      label: this._activeLabel,
      query: this._activeQuery,
    });
    this._activeQuery = null;
    this._activeLabel = null;
    this._passCount++;
  }

  /** Call once per frame after all passes. Resolves completed queries. */
  frameEnd() {
    if (!this.enabled) return;
    const gl = this.gl;

    // Try to resolve all pending queries
    const stillPending = [];
    for (const p of this._pending) {
      const available = gl.getQueryParameter(
        p.query,
        gl.QUERY_RESULT_AVAILABLE,
      );
      if (available) {
        const ns = gl.getQueryParameter(p.query, gl.QUERY_RESULT);
        gl.deleteQuery(p.query);
        const us = ns / 1000; // nanoseconds → microseconds
        const prev = this._currentFrame.get(p.label) || 0;
        this._currentFrame.set(p.label, prev + us);
      } else {
        stillPending.push(p);
      }
    }
    this._pending = stillPending;

    // Push current frame data to ring buffer
    if (this._currentFrame.size > 0) {
      this._frames.push(new Map(this._currentFrame));
      if (this._frames.length > this.ringSize) this._frames.shift();
    }
    this._currentFrame = new Map();

    this._passCountRing.push(this._passCount);
    if (this._passCountRing.length > this.ringSize)
      this._passCountRing.shift();
    this._passCount = 0;
  }

  /**
   * Get averaged timing summary over the ring buffer.
   * @returns {{ perPass: Map<string, number>, totalUs: number, passCount: number, frameCount: number }}
   */
  getSummary() {
    const nFrames = this._frames.length;
    if (nFrames === 0)
      return { perPass: new Map(), totalUs: 0, passCount: 0, frameCount: 0 };

    const sums = new Map();
    let totalSum = 0;
    for (const frame of this._frames) {
      for (const [label, us] of frame) {
        sums.set(label, (sums.get(label) || 0) + us);
        totalSum += us;
      }
    }

    const perPass = new Map();
    for (const [label, sum] of sums) {
      perPass.set(label, sum / nFrames);
    }

    const avgPassCount =
      this._passCountRing.length > 0
        ? this._passCountRing.reduce((a, b) => a + b, 0) /
          this._passCountRing.length
        : 0;

    return {
      perPass,
      totalUs: totalSum / nFrames,
      passCount: avgPassCount,
      frameCount: nFrames,
    };
  }

  /**
   * Format a human-readable summary string.
   * @returns {string}
   */
  formatSummary() {
    const s = this.getSummary();
    if (s.frameCount === 0) return "GPUProfiler: no data yet";
    const lines = [
      `GPU Timing (avg over ${s.frameCount} frames):`,
      `  Total: ${s.totalUs.toFixed(0)} µs/timestep`,
      `  Passes: ${s.passCount.toFixed(1)}/timestep`,
    ];
    const sorted = [...s.perPass.entries()].sort((a, b) => b[1] - a[1]);
    for (const [label, us] of sorted) {
      lines.push(`  ${label}: ${us.toFixed(0)} µs`);
    }
    return lines.join("\n");
  }

  /** Reset all collected data. */
  reset() {
    const gl = this.gl;
    this._pending.forEach((p) => gl.deleteQuery(p.query));
    this._pending = [];
    this._frames = [];
    this._currentFrame = new Map();
    this._passCountRing = [];
    this._passCount = 0;
  }

  /** Clean up all resources. */
  dispose() {
    this.reset();
    this.enabled = false;
  }
}
