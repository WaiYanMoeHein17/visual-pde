
// Each entry: { alpha, delta, beta, src1 (texture index for RHS state), dst (output texture) }
const SSPRK_STAGES = {
  2: [
    { alpha: 1, delta: 0, beta: 1, src1: 1, dst: 2 },
    { alpha: 0.5, delta: 0.5, beta: 0.5, src1: 2, dst: 0 },
  ],
  3: [
    { alpha: 1, delta: 0, beta: 1, src1: 1, dst: 2 },
    { alpha: 3 / 4, delta: 1 / 4, beta: 1 / 4, src1: 2, dst: 3 },
    { alpha: 1 / 3, delta: 2 / 3, beta: 2 / 3, src1: 3, dst: 0 },
  ],
};

// Classical RK4: stages 1-3 produce complete states, stage 4 is a special corrector.
const RK4_STAGES = [
  { alpha: 1, delta: 0, beta: 0.5, src1: 1, dst: 2 }, // y₂ = u^n + ½dt*F(u^n)
  { alpha: 1, delta: 0, beta: 0.5, src1: 2, dst: 3 }, // y₃ = u^n + ½dt*F(y₂)
  { alpha: 1, delta: 0, beta: 1, src1: 3, dst: 4 }, // y₄ = u^n + dt*F(y₃)
  "RK4_FINAL", // u^{n+1} = (-u^n + y₂ + 2y₃ + y₄)/3 + (dt/6)*F(y₄)
];

export function aderDgTimestep(
  simDomain,
  simMaterials,
  uniforms,
  simTextures,
  renderer,
  simScene,
  simCamera,
  options,
) {
  const dt = options.dt;
  uniforms.dt.value = dt;

  let order = parseInt(options.dgOrder);
  if (!Number.isFinite(order)) order = 2;
  order = Math.min(Math.max(order, 1), 4);

  // At DG element boundaries, nodes are duplicated (the rightmost node of
  // element i occupies the same position as the leftmost node of element i+1).
  // After each timestep we average these shared nodes for continuity.
  const applyBoundaryAvg = () => {
    if (simMaterials.DGBoundaryAvg) {
      simDomain.material = simMaterials.DGBoundaryAvg;
      uniforms.textureSource.value = simTextures[1].texture;
      renderer.setRenderTarget(simTextures[0]);
      renderer.render(simScene, simCamera);
      simTextures.rotate(-1);
    }
  };

  if (order === 1) {
    // Order 1: Forward Euler — single render pass, no stages needed.
    simDomain.material = simMaterials.DGFE;
    uniforms.textureSource.value = simTextures[1].texture;
    renderer.setRenderTarget(simTextures[0]);
    renderer.render(simScene, simCamera);
    simTextures.rotate(-1);
    applyBoundaryAvg();
    uniforms.t.value += dt;
  } else if (order <= 3) {
    // SSPRK2 or SSPRK3: loop through Shu-Osher stages.
    const stages = SSPRK_STAGES[order];
    for (const stage of stages) {
      simDomain.material = simMaterials.DGStage;
      uniforms.stageAlpha.value = stage.alpha;
      uniforms.stageDelta.value = stage.delta;
      uniforms.stageBeta.value = stage.beta;
      uniforms.textureSource.value = simTextures[1].texture; // u^n
      uniforms.textureSource1.value = simTextures[stage.src1].texture;
      renderer.setRenderTarget(simTextures[stage.dst]);
      renderer.render(simScene, simCamera);
    }
    // Final stage wrote to tex[0]; rotate so tex[1] = new state.
    simTextures.rotate(-1);
    applyBoundaryAvg();
    uniforms.t.value += dt;
  } else {
    // Order 4: Classical RK4 with complete state storage.
    for (const stage of RK4_STAGES) {
      if (stage === "RK4_FINAL") {
        // Final corrector: reads u^n, y₂, y₃, y₄ and evaluates F(y₄).
        simDomain.material = simMaterials.DGRK4Corr;
        uniforms.textureSource.value = simTextures[1].texture; // u^n
        uniforms.textureSource1.value = simTextures[4].texture; // y₄
        uniforms.textureSource2.value = simTextures[2].texture; // y₂
        uniforms.textureSource3.value = simTextures[3].texture; // y₃
        renderer.setRenderTarget(simTextures[0]);
        renderer.render(simScene, simCamera);
      } else {
        // Generic Shu-Osher stage.
        simDomain.material = simMaterials.DGStage;
        uniforms.stageAlpha.value = stage.alpha;
        uniforms.stageDelta.value = stage.delta;
        uniforms.stageBeta.value = stage.beta;
        uniforms.textureSource.value = simTextures[1].texture; // u^n
        uniforms.textureSource1.value = simTextures[stage.src1].texture;
        renderer.setRenderTarget(simTextures[stage.dst]);
        renderer.render(simScene, simCamera);
      }
    }
    simTextures.rotate(-1);
    applyBoundaryAvg();
    uniforms.t.value += dt;
  }
}
