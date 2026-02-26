/**
 * True ADER-DG timestepping with element-local predictor and single-step corrector.
 *
 * Unlike the previous RK-DG hybrid (SSPRK + RK4) in ader_dg_old_ssprk.js,
 * this implements a genuine ADER (Arbitrary high-order DERivatives) approach:
 *
 *   1. PREDICTOR (element-local): Picard iteration using only intra-element
 *      spatial derivatives. At element boundaries the predictor applies
 *      zero-gradient (no inter-element numerical flux). This builds a
 *      high-order temporal prediction within each element.
 *
 *   2. CORRECTOR (single step, full flux): Uses the predictor values at
 *      Gauss-Legendre quadrature points in [0, dt] together with the full
 *      DG spatial operator (including inter-element numerical flux) to
 *      compute the final update in a single step.
 *
 * Why this differs from the old SSPRK/RK4 hybrid:
 *   - In RK-DG, every stage evaluates the full spatial operator including
 *     inter-element numerical fluxes. This means p stages = p global
 *     communication steps.
 *   - In true ADER-DG, the predictor is purely element-local (no inter-element
 *     flux), and only the corrector performs one global communication step.
 *     The predictor builds temporal accuracy through Picard iteration using
 *     element-internal spatial derivatives only.
 *
 * Temporal quadrature:
 *   Order 1 : Forward Euler (no quadrature needed)
 *   Order 2 : 1-point midpoint rule  (exact for degree <= 1)
 *   Orders 3-4: 2-point Gauss-Legendre (exact for degree <= 3)
 *
 * Number of Picard iterations determines predictor accuracy:
 *   k iterations -> predictor error O(dt^{k+1})
 *   Combined with p-point GL quadrature (exact up to degree 2p-1):
 *     overall method order = min(k+1, 2p)
 *
 * Texture layout (non-fused):
 *   tex[1]       - u^n (read-only during timestep)
 *   tex[2]-[4]   - predictor states at temporal quadrature points
 *   tex[0]       - output buffer (before rotation)
 *
 * Texture layout (fused, orders 3-4):
 *   tex[1]       - u^n (read-only during timestep)
 *   mrtTargets   - pingpong MRT pair holding predictor states
 *   tex[0]       - output buffer (before rotation)
 *
 * Pass count per timestep:
 *   Non-fused (original):
 *     Order 1:  2  (1 Euler + boundary avg)
 *     Order 2:  3  (1 predictor + 1 corrector + boundary avg)
 *     Order 3:  6  (2x2 predictors + 1 corrector + boundary avg)
 *     Order 4:  8  (3x2 predictors + 1 corrector + boundary avg)
 *
 *   Fused (kernel fusion enabled):
 *     Order 1:  1  (fused FE + boundary avg)
 *     Order 2:  2  (1 predictor + fused corrector)
 *     Order 3:  3  (MRT pred iter1 + MRT pred iter2 + fused corrector)
 *     Order 4:  4  (MRT pred x3 + fused corrector)
 */

// ────────────────────────────────────────────────────────────────────────
// ADER-DG Gauss-Legendre temporal quadrature nodes & Picard weights
// ────────────────────────────────────────────────────────────────────────

// 2-point Gauss-Legendre nodes on [0, 1]:
//   c1 = (3 - sqrt(3)) / 6 ~ 0.21132,   c2 = (3 + sqrt(3)) / 6 ~ 0.78868
//   Quadrature weights: w1 = w2 = 1/2
const GL2_C1 = (3 - Math.sqrt(3)) / 6;
const GL2_C2 = (3 + Math.sqrt(3)) / 6;

// Picard integration weights a_{kj} = (1/dt) * integral_0^{c_k*dt} L_j(s) ds
// where L_j are Lagrange interpolants at tau_1 = c1*dt, tau_2 = c2*dt.
//
// a_{11} = 1/4,                           a_{12} = c1 - 1/4 = -(2*sqrt(3)-3)/12
// a_{21} = c2 - 1/4 = (2*sqrt(3)+3)/12,  a_{22} = 1/4
const PICARD_A11 = 0.25;
const PICARD_A12 = GL2_C1 - 0.25; // ~ -0.03868
const PICARD_A21 = GL2_C2 - 0.25; // ~  0.53868
const PICARD_A22 = 0.25;

export function aderDgTimestep(
  simDomain,
  simMaterials,
  uniforms,
  simTextures,
  renderer,
  simScene,
  simCamera,
  options,
  gpuProfiler,
  mrtTargets,
) {
  const dt = options.dt;
  uniforms.dt.value = dt;
  const fused = !!options.fusedKernels && !!mrtTargets;

  let order = parseInt(options.dgOrder);
  if (!Number.isFinite(order)) order = 2;
  order = Math.min(Math.max(order, 1), 4);

  // ── Profiler-wrapped render helper ───────────────────────────────────
  const renderPass = (label) => {
    if (gpuProfiler) gpuProfiler.beginPass(label);
    renderer.render(simScene, simCamera);
    if (gpuProfiler) gpuProfiler.endPass();
  };

  // ====================================================================
  //  Non-fused helpers (original algorithm)
  // ====================================================================

  // At DG element boundaries, nodes are duplicated (the rightmost node of
  // element i occupies the same position as the leftmost node of element i+1).
  // After each timestep we average these shared nodes for continuity.
  const applyBoundaryAvg = () => {
    if (simMaterials.DGBoundaryAvg) {
      simDomain.material = simMaterials.DGBoundaryAvg;
      uniforms.textureSource.value = simTextures[1].texture;
      renderer.setRenderTarget(simTextures[0]);
      renderPass("boundaryAvg");
      simTextures.rotate(-1);
    }
  };

  // ── Helper: run an element-local predictor pass ──────────────────────
  // Computes: dst = u^n + beta * dt * F_local(src)
  // Uses DGADERPred shader (zero-gradient at element boundaries).
  const predictLocal = (srcIdx, dstIdx, beta) => {
    simDomain.material = simMaterials.DGADERPred;
    uniforms.stageAlpha.value = 1.0;
    uniforms.stageDelta.value = 0.0;
    uniforms.stageBeta.value = beta;
    uniforms.textureSource.value = simTextures[1].texture; // u^n
    uniforms.textureSource1.value = simTextures[srcIdx].texture;
    renderer.setRenderTarget(simTextures[dstIdx]);
    renderPass("predictLocal");
  };

  // ── Helper: run a Picard iteration for 2-pt Gauss predictor ─────────
  // Computes: dst = u^n + w1*dt*F_local(src1) + w2*dt*F_local(src2)
  // Uses DGADERPicard2 shader (element-local, reads 2 source textures).
  const predictPicard2 = (src1Idx, src2Idx, dstIdx, w1, w2) => {
    simDomain.material = simMaterials.DGADERPicard2;
    uniforms.stageAlpha.value = 1.0; // weight for u^n
    uniforms.stageWeight1.value = w1;
    uniforms.stageWeight2.value = w2;
    uniforms.textureSource.value = simTextures[1].texture; // u^n
    uniforms.textureSource1.value = simTextures[src1Idx].texture;
    uniforms.textureSource2.value = simTextures[src2Idx].texture;
    renderer.setRenderTarget(simTextures[dstIdx]);
    renderPass("predictPicard2");
  };

  // ── Helper: run a single-source corrector pass (midpoint) ───────────
  // Computes: dst = u^n + beta * dt * F(src)
  // Uses DGStage shader (full inter-element flux).
  const correctSingle = (srcIdx, dstIdx, beta) => {
    simDomain.material = simMaterials.DGStage;
    uniforms.stageAlpha.value = 1.0;
    uniforms.stageDelta.value = 0.0;
    uniforms.stageBeta.value = beta;
    uniforms.textureSource.value = simTextures[1].texture; // u^n
    uniforms.textureSource1.value = simTextures[srcIdx].texture;
    renderer.setRenderTarget(simTextures[dstIdx]);
    renderPass("correctSingle");
  };

  // ── Helper: run a 2-point Gauss-Legendre corrector pass ─────────────
  // Computes: dst = u^n + dt/2 * [F(src1) + F(src2)]   (GL weights = 1/2)
  // Uses DGADERCorr2 shader (full flux, reads 2 source textures).
  const correct2pt = (src1Idx, src2Idx, dstIdx) => {
    simDomain.material = simMaterials.DGADERCorr2;
    uniforms.textureSource.value = simTextures[1].texture; // u^n
    uniforms.textureSource1.value = simTextures[src1Idx].texture;
    uniforms.textureSource2.value = simTextures[src2Idx].texture;
    renderer.setRenderTarget(simTextures[dstIdx]);
    renderPass("correct2pt");
  };

  // ====================================================================
  //  Fused / MRT helpers (kernel fusion path)
  // ====================================================================

  // ── MRT predictor: two element-local predictions in a single pass ───
  // Renders to an MRT target (2 color attachments).
  //   gl_FragData[0] = u^n + beta1 * dt * F_local(srcTex)
  //   gl_FragData[1] = u^n + beta2 * dt * F_local(srcTex)
  const predictLocalMRT = (srcTex, beta1, beta2, mrtDst) => {
    simDomain.material = simMaterials.DGADERPredMRT;
    uniforms.stageAlpha.value = 1.0;
    uniforms.stageDelta.value = 0.0;
    uniforms.stageBeta.value = beta1;
    uniforms.stageBeta2.value = beta2;
    uniforms.textureSource.value = simTextures[1].texture; // u^n
    uniforms.textureSource1.value = srcTex;
    renderer.setRenderTarget(mrtDst);
    renderPass("predictLocalMRT");
  };

  // ── MRT Picard2: two Picard iterations in a single pass ─────────────
  // Reads two source textures (tau_1 and tau_2 predictor states) and
  // writes two outputs with different weight pairs.
  //   gl_FragData[0] = u^n + (w1 *F(src1) + w2 *F(src2)) * dt
  //   gl_FragData[1] = u^n + (w1b*F(src1) + w2b*F(src2)) * dt
  const predictPicard2MRT = (src1Tex, src2Tex, mrtDst) => {
    simDomain.material = simMaterials.DGADERPicard2MRT;
    uniforms.stageAlpha.value = 1.0;
    uniforms.stageWeight1.value = PICARD_A11;
    uniforms.stageWeight2.value = PICARD_A12;
    uniforms.stageWeight1b.value = PICARD_A21;
    uniforms.stageWeight2b.value = PICARD_A22;
    uniforms.textureSource.value = simTextures[1].texture; // u^n
    uniforms.textureSource1.value = src1Tex;
    uniforms.textureSource2.value = src2Tex;
    renderer.setRenderTarget(mrtDst);
    renderPass("predictPicard2MRT");
  };

  // ── Fused corrector + boundary avg (2-pt GL, orders 3-4) ───────────
  // Reads two predictor textures (any source, including MRT attachments).
  const correct2ptFused = (src1Tex, src2Tex) => {
    simDomain.material = simMaterials.DGADERCorr2Fused;
    uniforms.textureSource.value = simTextures[1].texture; // u^n
    uniforms.textureSource1.value = src1Tex;
    uniforms.textureSource2.value = src2Tex;
    renderer.setRenderTarget(simTextures[0]);
    renderPass("correct2ptFused");
    simTextures.rotate(-1);
  };

  // ── Fused corrector + boundary avg (single source, order 2) ────────
  const correctSingleFused = (srcIdx) => {
    simDomain.material = simMaterials.DGStageFused;
    uniforms.stageAlpha.value = 1.0;
    uniforms.stageDelta.value = 0.0;
    uniforms.stageBeta.value = 1.0;
    uniforms.textureSource.value = simTextures[1].texture; // u^n
    uniforms.textureSource1.value = simTextures[srcIdx].texture;
    renderer.setRenderTarget(simTextures[0]);
    renderPass("correctSingleFused");
    simTextures.rotate(-1);
  };

  // ====================================================================
  //  ADER-DG timestepping
  // ====================================================================

  if (order === 1) {
    if (fused) {
      // ── Order 1 fused: 1 pass (FE + boundary avg inline) ───────────
      simDomain.material = simMaterials.DGFEFused;
      uniforms.textureSource.value = simTextures[1].texture;
      renderer.setRenderTarget(simTextures[0]);
      renderPass("FEFused");
      simTextures.rotate(-1);
    } else {
      // ── Order 1: Forward Euler + boundary avg ── 2 passes ──────────
      simDomain.material = simMaterials.DGFE;
      uniforms.textureSource.value = simTextures[1].texture;
      renderer.setRenderTarget(simTextures[0]);
      renderPass("FE");
      simTextures.rotate(-1);
      applyBoundaryAvg();
    }
    uniforms.t.value += dt;

  } else if (order === 2) {
    if (fused) {
      // ── Order 2 fused: 2 passes ────────────────────────────────────
      //   1. predictLocal -> tex[2]
      //   2. fused corrector + boundary avg -> tex[0], rotate
      predictLocal(1, 2, 0.5);
      correctSingleFused(2);
    } else {
      // ── Order 2: ADER midpoint predictor-corrector ── 3 passes ─────
      predictLocal(1, 2, 0.5);        // q at midpoint -> tex[2]
      correctSingle(2, 0, 1.0);       // u^{n+1} -> tex[0]
      simTextures.rotate(-1);
      applyBoundaryAvg();
    }
    uniforms.t.value += dt;

  } else if (order === 3) {
    if (fused) {
      // ── Order 3 fused: 3 passes ────────────────────────────────────
      //   1. MRT Picard iter 1: u^n -> mrtTargets[0]
      //      out[0] = q1(tau_1), out[1] = q1(tau_2)
      //   2. MRT Picard iter 2: mrtTargets[0] -> mrtTargets[1]
      //      out[0] = q2(tau_1), out[1] = q2(tau_2)
      //   3. Fused corrector: reads mrtTargets[1], writes tex[0]
      predictLocalMRT(simTextures[1].texture, GL2_C1, GL2_C2, mrtTargets[0]);
      predictPicard2MRT(
        mrtTargets[0].texture[0],
        mrtTargets[0].texture[1],
        mrtTargets[1],
      );
      correct2ptFused(mrtTargets[1].texture[0], mrtTargets[1].texture[1]);
    } else {
      // ── Order 3: ADER with 2-pt GL, 2 Picard iterations ── 6 passes
      predictLocal(1, 2, GL2_C1);        // q1(tau_1) -> tex[2]
      predictLocal(1, 3, GL2_C2);        // q1(tau_2) -> tex[3]
      predictPicard2(2, 3, 4, PICARD_A11, PICARD_A12);  // q2(tau_1) -> tex[4]
      predictPicard2(2, 3, 2, PICARD_A21, PICARD_A22);  // q2(tau_2) -> tex[2]
      correct2pt(4, 2, 0);              // u^{n+1} -> tex[0]
      simTextures.rotate(-1);
      applyBoundaryAvg();
    }
    uniforms.t.value += dt;

  } else {
    // order === 4
    if (fused) {
      // ── Order 4 fused: 4 passes ────────────────────────────────────
      //   1. MRT Picard iter 1: u^n -> mrtTargets[0]
      //   2. MRT Picard iter 2: mrtTargets[0] -> mrtTargets[1]
      //   3. MRT Picard iter 3: mrtTargets[1] -> mrtTargets[0]  (pingpong)
      //   4. Fused corrector: reads mrtTargets[0], writes tex[0]
      predictLocalMRT(simTextures[1].texture, GL2_C1, GL2_C2, mrtTargets[0]);
      predictPicard2MRT(
        mrtTargets[0].texture[0],
        mrtTargets[0].texture[1],
        mrtTargets[1],
      );
      predictPicard2MRT(
        mrtTargets[1].texture[0],
        mrtTargets[1].texture[1],
        mrtTargets[0],
      );
      correct2ptFused(mrtTargets[0].texture[0], mrtTargets[0].texture[1]);
    } else {
      // ── Order 4: ADER with 2-pt GL, 3 Picard iterations ── 8 passes
      predictLocal(1, 2, GL2_C1);        // q1(tau_1) -> tex[2]
      predictLocal(1, 3, GL2_C2);        // q1(tau_2) -> tex[3]
      predictPicard2(2, 3, 4, PICARD_A11, PICARD_A12);  // q2(tau_1) -> tex[4]
      predictPicard2(2, 3, 2, PICARD_A21, PICARD_A22);  // q2(tau_2) -> tex[2]
      predictPicard2(4, 2, 3, PICARD_A11, PICARD_A12);  // q3(tau_1) -> tex[3]
      predictPicard2(4, 2, 4, PICARD_A21, PICARD_A22);  // q3(tau_2) -> tex[4]
      correct2pt(3, 4, 0);              // u^{n+1} -> tex[0]
      simTextures.rotate(-1);
      applyBoundaryAvg();
    }
    uniforms.t.value += dt;
  }
}
