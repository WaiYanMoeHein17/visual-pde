// simulation_shaders.js

/**
 * Generates the top part of a shader for reaction-diffusion simulation based on the given timestepping scheme.
 * @param {string} type - The timestepping scheme to generate the shader for.
 * @returns {string} The generated shader code.
 */
export function RDShaderTop(type) {
  let numInputs = 0;
  switch (type) {
    case "FE":
      numInputs = 2;
      break;
    case "AB2":
      numInputs = 2;
      break;
    case "Mid1":
      numInputs = 1;
      break;
    case "Mid2":
      numInputs = 2;
      break;
    case "RK41":
      numInputs = 1;
      break;
    case "RK42":
      numInputs = 2;
      break;
    case "RK43":
      numInputs = 3;
      break;
    case "RK44":
      numInputs = 4;
      break;
    case "ADER":
      numInputs = 2;
      break;
  }
  let parts = [];
  parts[0] =
    "precision highp float; precision highp sampler2D; varying vec2 textureCoords;";
  parts[1] = "uniform sampler2D textureSource;";
  parts[2] = "uniform sampler2D textureSource1;";
  parts[3] = "uniform sampler2D textureSource2;";
  parts[4] = "uniform sampler2D textureSource3;";
  return (
    parts.slice(0, numInputs + 1).join("\n") +
    `
    uniform float dt;
    uniform float dx;
    uniform float dy;
    uniform float L;
    uniform float L_x;
    uniform float L_y;
    uniform float L_min;
    uniform float t;
    uniform float seed;
    uniform sampler2D imageSourceOne;
    uniform sampler2D imageSourceTwo;

    AUXILIARY_GLSL_FUNS

    const float ALPHA = 0.147;
    const float INV_ALPHA = 1.0 / ALPHA;
    const float BETA = 2.0 / (pi * ALPHA);
    float erfinv(float pERF) {
      float yERF;
      if (pERF == -1.0) {
        yERF = log(1.0 - (-0.99)*(-0.99));
      } else {
        yERF = log(1.0 - pERF*pERF);
      }
      float zERF = BETA + 0.5 * yERF;
      return sqrt(sqrt(zERF*zERF - yERF * INV_ALPHA) - zERF) * sign(pERF);
    }

    void computeRHS(sampler2D textureSource, vec4 uvwqIn, vec4 uvwqLIn, vec4 uvwqRIn, vec4 uvwqTIn, vec4 uvwqBIn, vec4 uvwqLLIn, vec4 uvwqRRIn, vec4 uvwqTTIn, vec4 uvwqBBIn, out highp vec4 result) {

        ivec2 texSize = textureSize(textureSource,0);
        float step_x = 1.0 / float(texSize.x);
        float step_y = 1.0 / float(texSize.y);
        float x = textureCoords.x * L_x + MINX;
        float y = textureCoords.y * L_y + MINY;
        float interior = float(textureCoords.x > 0.75*step_x && textureCoords.x < 1.0 - 0.75*step_x && textureCoords.y > 0.5*step_y && textureCoords.y < 1.0 - 0.75*step_y);
        float exterior = 1.0 - interior;
        vec2 dSquared = 1.0/vec2(dx*dx, dy*dy);
        vec2 textureCoordsL = textureCoords + vec2(-step_x, 0.0);
        vec2 textureCoordsLL = textureCoordsL + vec2(-step_x, 0.0);
        vec2 textureCoordsR = textureCoords + vec2(+step_x, 0.0);
        vec2 textureCoordsRR = textureCoordsR + vec2(+step_x, 0.0);
        vec2 textureCoordsT = textureCoords + vec2(0.0, +step_y);
        vec2 textureCoordsTT = textureCoordsT + vec2(0.0, +step_y);
        vec2 textureCoordsB = textureCoords + vec2(0.0, -step_y);
        vec2 textureCoordsBB = textureCoordsB + vec2(0.0, -step_y);

        vec4 uvwq = uvwqIn;
        vec4 uvwqL = uvwqLIn;
        vec4 uvwqLL = uvwqLLIn;
        vec4 uvwqR = uvwqRIn;
        vec4 uvwqRR = uvwqRRIn;
        vec4 uvwqT = uvwqTIn;
        vec4 uvwqTT = uvwqTTIn;
        vec4 uvwqB = uvwqBIn;
        vec4 uvwqBB = uvwqBBIn;
    `
  );
}

/**
 * Generates the top part of a shader for ADER-DG timestepping.
 * @returns {string} The generated shader code.
 */
export function RDShaderTopDG() {
  return `
    precision highp float; precision highp sampler2D; varying vec2 textureCoords;
    uniform sampler2D textureSource;
    uniform sampler2D textureSource1;
    uniform sampler2D textureSource2;
    uniform sampler2D textureSource3;
    uniform float dt;
    uniform float dx;
    uniform float dy;
    uniform float L;
    uniform float L_x;
    uniform float L_y;
    uniform float L_min;
    uniform float t;
    uniform float seed;
    uniform float dgOrder;
    uniform float stageAlpha;
    uniform float stageDelta;
    uniform float stageBeta;
    uniform float stageWeight1;
    uniform float stageWeight2;
    uniform float stageBeta2;
    uniform float stageWeight1b;
    uniform float stageWeight2b;
    uniform sampler2D imageSourceOne;
    uniform sampler2D imageSourceTwo;

    AUXILIARY_GLSL_FUNS

    const float ALPHA = 0.147;
    const float INV_ALPHA = 1.0 / ALPHA;
    const float BETA = 2.0 / (pi * ALPHA);
    float erfinv(float pERF) {
      float yERF;
      if (pERF == -1.0) {
        yERF = log(1.0 - (-0.99)*(-0.99));
      } else {
        yERF = log(1.0 - pERF*pERF);
      }
      float zERF = BETA + 0.5 * yERF;
      return sqrt(sqrt(zERF*zERF - yERF * INV_ALPHA) - zERF) * sign(pERF);
    }

    int dgWrapIndex(int idx, int n) {
      int m = idx % n;
      if (m < 0) {
        m += n;
      }
      return m;
    }

    int dgStepLeft(int idx, int order, int n) {
      int nodesPerElem = order + 1;
      int elem = idx / nodesPerElem;
      int local = idx - elem * nodesPerElem;
      if (local == 0) {
        int prevElem = elem - 1;
        int raw = prevElem * nodesPerElem + (order - 1);
        return dgWrapIndex(raw, n);
      }
      return dgWrapIndex(idx - 1, n);
    }

    int dgStepRight(int idx, int order, int n) {
      int nodesPerElem = order + 1;
      int elem = idx / nodesPerElem;
      int local = idx - elem * nodesPerElem;
      if (local == order) {
        int nextElem = elem + 1;
        int raw = nextElem * nodesPerElem + 1;
        return dgWrapIndex(raw, n);
      }
      return dgWrapIndex(idx + 1, n);
    }

    int dgStepDown(int idy, int order, int n) {
      int nodesPerElem = order + 1;
      int elem = idy / nodesPerElem;
      int local = idy - elem * nodesPerElem;
      if (local == 0) {
        int prevElem = elem - 1;
        int raw = prevElem * nodesPerElem + (order - 1);
        return dgWrapIndex(raw, n);
      }
      return dgWrapIndex(idy - 1, n);
    }

    int dgStepUp(int idy, int order, int n) {
      int nodesPerElem = order + 1;
      int elem = idy / nodesPerElem;
      int local = idy - elem * nodesPerElem;
      if (local == order) {
        int nextElem = elem + 1;
        int raw = nextElem * nodesPerElem + 1;
        return dgWrapIndex(raw, n);
      }
      return dgWrapIndex(idy + 1, n);
    }

    vec2 dgTexCoord(int idx, float step_x) {
      return vec2((float(idx) + 0.5) * step_x, textureCoords.y);
    }

    vec2 dgTexCoord2D(int idx, int idy, float step_x, float step_y) {
      return vec2((float(idx) + 0.5) * step_x, (float(idy) + 0.5) * step_y);
    }

    vec4 dgSample(sampler2D tex, int idx, float step_x, int texWidth) {
      int wrapped = dgWrapIndex(idx, texWidth);
      return texture2D(tex, dgTexCoord(wrapped, step_x));
    }

    vec4 dgSample2D(sampler2D tex, int idx, int idy, float step_x, float step_y, int texWidth, int texHeight) {
      int wrappedX = dgWrapIndex(idx, texWidth);
      int wrappedY = dgWrapIndex(idy, texHeight);
      return texture2D(tex, dgTexCoord2D(wrappedX, wrappedY, step_x, step_y));
    }

    vec4 dgElementAverage(
      sampler2D tex,
      int elem,
      int order,
      int nodesPerElem,
      int texWidth,
      float step_x
    ) {
      vec4 sum = vec4(0.0);
      for (int i = 0; i < 5; i++) {
        if (i <= order) {
          int idx = elem * nodesPerElem + i;
          sum += dgSample(tex, idx, step_x, texWidth);
        }
      }
      return sum / float(nodesPerElem);
    }

    void dgElementMinMax(
      sampler2D tex,
      int elem,
      int order,
      int nodesPerElem,
      int texWidth,
      float step_x,
      out vec4 minVal,
      out vec4 maxVal
    ) {
      minVal = vec4(1.0e20);
      maxVal = vec4(-1.0e20);
      for (int i = 0; i < 5; i++) {
        if (i <= order) {
          int idx = elem * nodesPerElem + i;
          vec4 v = dgSample(tex, idx, step_x, texWidth);
          minVal = min(minVal, v);
          maxVal = max(maxVal, v);
        }
      }
    }
    `
  ;
}

/**
 * Generates a shader that enforces continuity at DG element boundaries by averaging
 * the values at shared boundary nodes.
 */
export function RDShaderDGBoundaryAvg() {
  return `
  varying vec2 textureCoords;
  uniform sampler2D textureSource;
  uniform float dgOrder;

  void main() {
    ivec2 texSize = textureSize(textureSource, 0);
    int texWidth = texSize.x;
    int texHeight = texSize.y;
    float step_x = 1.0 / float(texWidth);
    float step_y = 1.0 / float(texHeight);

    int order = int(dgOrder + 0.5);
    int nodesPerElem = order + 1;

    int ix = int(floor(textureCoords.x * float(texWidth)));
    int iy = int(floor(textureCoords.y * float(texHeight)));

    int elem = ix / nodesPerElem;
    int local_node = ix - elem * nodesPerElem;
    int nElem = texWidth / nodesPerElem;

    vec4 current = texture2D(textureSource, textureCoords);

    // At rightmost node of element: average with leftmost node of next element
    if (local_node == order) {
      int nextElem = (elem + 1) % nElem;
      int neighborIdx = nextElem * nodesPerElem;
      vec2 neighborCoord = vec2((float(neighborIdx) + 0.5) * step_x, (float(iy) + 0.5) * step_y);
      vec4 neighbor = texture2D(textureSource, neighborCoord);
      gl_FragColor = 0.5 * (current + neighbor);
    }
    // At leftmost node of element: average with rightmost node of previous element
    else if (local_node == 0) {
      int prevElem = (elem - 1 + nElem) % nElem;
      int neighborIdx = prevElem * nodesPerElem + order;
      vec2 neighborCoord = vec2((float(neighborIdx) + 0.5) * step_x, (float(iy) + 0.5) * step_y);
      vec4 neighbor = texture2D(textureSource, neighborCoord);
      gl_FragColor = 0.5 * (current + neighbor);
    }
    // Interior node: pass through unchanged
    else {
      gl_FragColor = current;
    }
  }
  `;
}

/**
 * Generates shader code for DG RHS computation.
 * This creates a computeRHS function that wraps the reaction-diffusion calculations
 * for use in DG timestepping.
 * @param {number} numSpecies - Number of species (1-4)
 * @param {string} reactionStrings - The parsed reaction strings (UFUN, VFUN, etc.)
 * @returns {string} - The shader code for the computeRHS function
 */
export function RDShaderDGComputeRHS(numSpecies, reactionStrings, diffusionStrings, crossDiffusion) {
  if (numSpecies == undefined) numSpecies = 4;
  if (reactionStrings == undefined) reactionStrings = "float UFUN = 0.0;\nfloat VFUN = 0.0;\nfloat WFUN = 0.0;\nfloat QFUN = 0.0;\n";
  if (diffusionStrings == undefined) diffusionStrings = "";
  if (crossDiffusion == undefined) crossDiffusion = false;
  
  // Build the RHS computation for DG
  // The DG method uses the passed-in neighbor values directly
  let shader = `
void computeRHS(sampler2D textureSource, vec4 uvwqIn, vec4 uvwqLIn, vec4 uvwqRIn, vec4 uvwqTIn, vec4 uvwqBIn, vec4 uvwqLLIn, vec4 uvwqRRIn, vec4 uvwqTTIn, vec4 uvwqBBIn, out highp vec4 result) {
    // Use passed-in values for state and neighbors
    vec4 uvwq = uvwqIn;
    vec4 uvwqL = uvwqLIn;
    vec4 uvwqLL = uvwqLLIn;
    vec4 uvwqR = uvwqRIn;
    vec4 uvwqRR = uvwqRRIn;
    vec4 uvwqT = uvwqTIn;
    vec4 uvwqTT = uvwqTTIn;
    vec4 uvwqB = uvwqBIn;
    vec4 uvwqBB = uvwqBBIn;
    
    // Access species values for use in reaction terms
    float u = uvwq.r;
    float v = uvwq.g;
    float w = uvwq.b;
    float q = uvwq.a;
    
    // Compute coordinate info using texture size
    ivec2 texSize = textureSize(textureSource, 0);
    float step_x_local = 1.0 / float(texSize.x);
    float step_y_local = 1.0 / float(texSize.y);
    int order = int(dgOrder + 0.5);
    int nodesPerElem = order + 1;
    int ix = int(floor(textureCoords.x * float(texSize.x)));
    int elem = ix / nodesPerElem;
    int local_node = ix - elem * nodesPerElem;
    float x = MINX + float(elem * order + local_node) * dx;
    float y = textureCoords.y * L_y + MINY;
    
    // Compute spatial derivatives as vec4 (for compatibility with parseReactionStrings)
    vec4 uvwqX = (uvwqR - uvwqL) / (2.0 * dx);
    vec4 uvwqY = (uvwqT - uvwqB) / (2.0 * dy);
    vec4 uvwqXF = (uvwqR - uvwq) / dx;
    vec4 uvwqXB = (uvwq - uvwqL) / dx;
    vec4 uvwqYF = (uvwqT - uvwq) / dy;
    vec4 uvwqYB = (uvwq - uvwqB) / dy;
    vec4 uvwqXX = (uvwqR - 2.0*uvwq + uvwqL) / (dx*dx);
    vec4 uvwqYY = (uvwqT - 2.0*uvwq + uvwqB) / (dy*dy);

    // Individual float derivatives for convenience
    float u_x = uvwqX.r;
    float u_y = uvwqY.r;
    float v_x = uvwqX.g;
    float v_y = uvwqY.g;
    float w_x = uvwqX.b;
    float w_y = uvwqY.b;
    float q_x = uvwqX.a;
    float q_y = uvwqY.a;

    // Second derivatives
    float u_xx = uvwqXX.r;
    float u_yy = uvwqYY.r;
    float v_xx = uvwqXX.g;
    float v_yy = uvwqYY.g;
    float w_xx = uvwqXX.b;
    float w_yy = uvwqYY.b;
    float q_xx = uvwqXX.a;
    float q_yy = uvwqYY.a;

    // Laplacians
    float Lapu = u_xx + u_yy;
    float Lapv = v_xx + v_yy;
    float Lapw = w_xx + w_yy;
    float Lapq = q_xx + q_yy;
    
    // Diffusion coefficient declarations
    ${diffusionStrings}

    // Include the reaction strings (defines UFUN, VFUN, WFUN, QFUN)
    ${reactionStrings}
`;

  // Compute du, dv, dw, dq with diffusion contributions
  if (!crossDiffusion) {
    // Normal diffusion
    shader += `    float LDuuU = 0.5*((Duux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DuuxR*(uvwqR.r - uvwq.r) + DuuxL*(uvwqL.r - uvwq.r)) / dx) / dx + 0.5*((Duuy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DuuyT*(uvwqT.r - uvwq.r) + DuuyB*(uvwqB.r - uvwq.r)) / dy) / dy;\n`;
    shader += `    float du = LDuuU + UFUN;\n`;

    if (numSpecies > 1) {
      shader += `    float LDvvV = 0.5*((Dvvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DvvxR*(uvwqR.g - uvwq.g) + DvvxL*(uvwqL.g - uvwq.g)) / dx) / dx + 0.5*((Dvvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DvvyT*(uvwqT.g - uvwq.g) + DvvyB*(uvwqB.g - uvwq.g)) / dy) / dy;\n`;
      shader += `    float dv = LDvvV + VFUN;\n`;
    }

    if (numSpecies > 2) {
      shader += `    float LDwwW = 0.5*((Dwwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DwwxR*(uvwqR.b - uvwq.b) + DwwxL*(uvwqL.b - uvwq.b)) / dx) / dx + 0.5*((Dwwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DwwyT*(uvwqT.b - uvwq.b) + DwwyB*(uvwqB.b - uvwq.b)) / dy) / dy;\n`;
      shader += `    float dw = LDwwW + WFUN;\n`;
    }

    if (numSpecies > 3) {
      shader += `    float LDqqQ = 0.5*((Dqqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DqqxR*(uvwqR.a - uvwq.a) + DqqxL*(uvwqL.a - uvwq.a)) / dx) / dx + 0.5*((Dqqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DqqyT*(uvwqT.a - uvwq.a) + DqqyB*(uvwqB.a - uvwq.a)) / dy) / dy;\n`;
      shader += `    float dq = LDqqQ + QFUN;\n`;
    }
  } else {
    // Cross-diffusion
    shader += `    vec2 dSquared = 1.0/vec2(dx*dx, dy*dy);\n`;

    shader += [
      `vec2 LDuuU = vec2(Duux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DuuxR*(uvwqR.r - uvwq.r) + DuuxL*(uvwqL.r - uvwq.r), Duuy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DuuyT*(uvwqT.r - uvwq.r) + DuuyB*(uvwqB.r - uvwq.r));`,
      `vec2 LDuvV = vec2(Duvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DuvxR*(uvwqR.g - uvwq.g) + DuvxL*(uvwqL.g - uvwq.g), Duvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DuvyT*(uvwqT.g - uvwq.g) + DuvyB*(uvwqB.g - uvwq.g));`,
      `vec2 LDuwW = vec2(Duwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DuwxR*(uvwqR.b - uvwq.b) + DuwxL*(uvwqL.b - uvwq.b), Duwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DuwyT*(uvwqT.b - uvwq.b) + DuwyB*(uvwqB.b - uvwq.b));`,
      `vec2 LDuqQ = vec2(Duqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DuqxR*(uvwqR.a - uvwq.a) + DuqxL*(uvwqL.a - uvwq.a), Duqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DuqyT*(uvwqT.a - uvwq.a) + DuqyB*(uvwqB.a - uvwq.a));`,
    ].slice(0, numSpecies).join("\n    ") +
    `\n    float du = 0.5*dot(dSquared,` +
    [`LDuuU`, `LDuvV`, `LDuwW`, `LDuqQ`].slice(0, numSpecies).join(" + ") +
    `) + UFUN;\n`;

    if (numSpecies > 1) {
      shader += [
        `    vec2 LDvuU = vec2(Dvux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DvuxR*(uvwqR.r - uvwq.r) + DvuxL*(uvwqL.r - uvwq.r), Dvuy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DvuyT*(uvwqT.r - uvwq.r) + DvuyB*(uvwqB.r - uvwq.r));`,
        `vec2 LDvvV = vec2(Dvvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DvvxR*(uvwqR.g - uvwq.g) + DvvxL*(uvwqL.g - uvwq.g), Dvvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DvvyT*(uvwqT.g - uvwq.g) + DvvyB*(uvwqB.g - uvwq.g));`,
        `vec2 LDvwW = vec2(Dvwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DvwxR*(uvwqR.b - uvwq.b) + DvwxL*(uvwqL.b - uvwq.b), Dvwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DvwyT*(uvwqT.b - uvwq.b) + DvwyB*(uvwqB.b - uvwq.b));`,
        `vec2 LDvqQ = vec2(Dvqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DvqxR*(uvwqR.a - uvwq.a) + DvqxL*(uvwqL.a - uvwq.a), Dvqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DvqyT*(uvwqT.a - uvwq.a) + DvqyB*(uvwqB.a - uvwq.a));`,
      ].slice(0, numSpecies).join("\n    ") +
      `\n    float dv = 0.5*dot(dSquared,` +
      [`LDvuU`, `LDvvV`, `LDvwW`, `LDvqQ`].slice(0, numSpecies).join(" + ") +
      `) + VFUN;\n`;
    }

    if (numSpecies > 2) {
      shader += [
        `    vec2 LDwuU = vec2(Dwux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DwuxR*(uvwqR.r - uvwq.r) + DwuxL*(uvwqL.r - uvwq.r), Dwuy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DwuyT*(uvwqT.r - uvwq.r) + DwuyB*(uvwqB.r - uvwq.r));`,
        `vec2 LDwvV = vec2(Dwvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DwvxR*(uvwqR.g - uvwq.g) + DwvxL*(uvwqL.g - uvwq.g), Dwvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DwvyT*(uvwqT.g - uvwq.g) + DwvyB*(uvwqB.g - uvwq.g));`,
        `vec2 LDwwW = vec2(Dwwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DwwxR*(uvwqR.b - uvwq.b) + DwwxL*(uvwqL.b - uvwq.b), Dwwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DwwyT*(uvwqT.b - uvwq.b) + DwwyB*(uvwqB.b - uvwq.b));`,
        `vec2 LDwqQ = vec2(Dwqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DwqxR*(uvwqR.a - uvwq.a) + DwqxL*(uvwqL.a - uvwq.a), Dwqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DwqyT*(uvwqT.a - uvwq.a) + DwqyB*(uvwqB.a - uvwq.a));`,
      ].slice(0, numSpecies).join("\n    ") +
      `\n    float dw = 0.5*dot(dSquared,` +
      [`LDwuU`, `LDwvV`, `LDwwW`, `LDwqQ`].slice(0, numSpecies).join(" + ") +
      `) + WFUN;\n`;
    }

    if (numSpecies > 3) {
      shader += [
        `    vec2 LDquU = vec2(Dqux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DquxR*(uvwqR.r - uvwq.r) + DquxL*(uvwqL.r - uvwq.r), Dquy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DquyT*(uvwqT.r - uvwq.r) + DquyB*(uvwqB.r - uvwq.r));`,
        `vec2 LDqvV = vec2(Dqvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DqvxR*(uvwqR.g - uvwq.g) + DqvxL*(uvwqL.g - uvwq.g), Dqvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DqvyT*(uvwqT.g - uvwq.g) + DqvyB*(uvwqB.g - uvwq.g));`,
        `vec2 LDqwW = vec2(Dqwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DqwxR*(uvwqR.b - uvwq.b) + DqwxL*(uvwqL.b - uvwq.b), Dqwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DqwyT*(uvwqT.b - uvwq.b) + DqwyB*(uvwqB.b - uvwq.b));`,
        `vec2 LDqqQ = vec2(Dqqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DqqxR*(uvwqR.a - uvwq.a) + DqqxL*(uvwqL.a - uvwq.a), Dqqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DqqyT*(uvwqT.a - uvwq.a) + DqqyB*(uvwqB.a - uvwq.a));`,
      ].slice(0, numSpecies).join("\n    ") +
      `\n    float dq = 0.5*dot(dSquared,` +
      [`LDquU`, `LDqvV`, `LDqwW`, `LDqqQ`].slice(0, numSpecies).join(" + ") +
      `) + QFUN;\n`;
    }
  }
  
  // Set the output based on number of species
  switch (numSpecies) {
    case 1:
      shader += `    result = vec4(du, 0.0, 0.0, 0.0);\n`;
      break;
    case 2:
      shader += `    result = vec4(du, dv, 0.0, 0.0);\n`;
      break;
    case 3:
      shader += `    result = vec4(du, dv, dw, 0.0);\n`;
      break;
    case 4:
      shader += `    result = vec4(du, dv, dw, dq);\n`;
      break;
  }
  
  shader += `}\n`;
  return shader;
}

/**
 * Generates shader code based on the timestepping scheme.
 * @param {string} type - The type of timestepping scheme ("FE", "AB2", "Mid1", "Mid2", "RK41", "RK42", "RK43", "RK44").
 * @returns {string} - The generated shader code.
 */
export function RDShaderMain(type) {
  let update = {};
  update.FE = `uvwq = texture2D(textureSource, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = dt * RHS / timescales + uvwq;`;
  update.AB2 = `uvwq = texture2D(textureSource, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS1);
    uvwq = texture2D(textureSource1, textureCoords);
    uvwqL = texture2D(textureSource1, textureCoordsL);
    uvwqR = texture2D(textureSource1, textureCoordsR);
    uvwqT = texture2D(textureSource1, textureCoordsT);
    uvwqB = texture2D(textureSource1, textureCoordsB);
    uvwqLL = texture2D(textureSource1, textureCoordsLL);
    uvwqRR = texture2D(textureSource1, textureCoordsRR);
    uvwqTT = texture2D(textureSource1, textureCoordsTT);
    uvwqBB = texture2D(textureSource1, textureCoordsBB);
    computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS2);
    RHS = 1.5 * RHS1 - 0.5 * RHS2;
    vec4 timescales = TIMESCALES;
    updated = dt * RHS / timescales + texture2D(textureSource, textureCoords);`;
  update.Mid1 = `uvwq = texture2D(textureSource, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = RHS;`;
  update.Mid2 = `uvwqLast = texture2D(textureSource, textureCoords);
    uvwq = uvwqLast + 0.5*dt*texture2D(textureSource1, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL) + 0.5*dt*texture2D(textureSource1, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR) + 0.5*dt*texture2D(textureSource1, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT) + 0.5*dt*texture2D(textureSource1, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB) + 0.5*dt*texture2D(textureSource1, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL) + 0.5*dt*texture2D(textureSource1, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR) + 0.5*dt*texture2D(textureSource1, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT) + 0.5*dt*texture2D(textureSource1, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB) + 0.5*dt*texture2D(textureSource1, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = dt * RHS / timescales + uvwqLast;`;
  update.RK41 = `uvwq = texture2D(textureSource, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = RHS;`;
  update.RK42 = `uvwq = texture2D(textureSource, textureCoords) + 0.5*dt*texture2D(textureSource1, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL) + 0.5*dt*texture2D(textureSource1, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR) + 0.5*dt*texture2D(textureSource1, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT) + 0.5*dt*texture2D(textureSource1, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB) + 0.5*dt*texture2D(textureSource1, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL) + 0.5*dt*texture2D(textureSource1, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR) + 0.5*dt*texture2D(textureSource1, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT) + 0.5*dt*texture2D(textureSource1, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB) + 0.5*dt*texture2D(textureSource1, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = RHS;`;
  update.RK43 = `uvwq = texture2D(textureSource, textureCoords) + 0.5*dt*texture2D(textureSource2, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL) + 0.5*dt*texture2D(textureSource2, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR) + 0.5*dt*texture2D(textureSource2, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT) + 0.5*dt*texture2D(textureSource2, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB) + 0.5*dt*texture2D(textureSource2, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL) + 0.5*dt*texture2D(textureSource2, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR) + 0.5*dt*texture2D(textureSource2, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT) + 0.5*dt*texture2D(textureSource2, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB) + 0.5*dt*texture2D(textureSource2, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = RHS;`;
  update.RK44 = `uvwqLast = texture2D(textureSource, textureCoords);
    uvwq = uvwqLast + dt*texture2D(textureSource3, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL) + dt*texture2D(textureSource3, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR) + dt*texture2D(textureSource3, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT) + dt*texture2D(textureSource3, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB) + dt*texture2D(textureSource3, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL) + dt*texture2D(textureSource3, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR) + dt*texture2D(textureSource3, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT) + dt*texture2D(textureSource3, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB) + dt*texture2D(textureSource3, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS1);
    RHS = (texture2D(textureSource1, textureCoords) + 2.0*(texture2D(textureSource2, textureCoords) + texture2D(textureSource3, textureCoords)) + RHS1) / 6.0;
    vec4 timescales = TIMESCALES;
    updated = dt * RHS / timescales + uvwqLast;`;
  update.ADER = `uvwq = texture2D(textureSource, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = dt * RHS / timescales + uvwq;`;
  return (
    `
  void main()
  {
      ivec2 texSize = textureSize(textureSource,0);
      float step_x = 1.0 / float(texSize.x);
      float step_y = 1.0 / float(texSize.y);
      float x = textureCoords.x * L_x + MINX;
      float y = textureCoords.y * L_y + MINY;
      float interior = float(textureCoords.x > 0.75*step_x && textureCoords.x < 1.0 - 0.75*step_x && textureCoords.y > 0.5*step_y && textureCoords.y < 1.0 - 0.75*step_y);
      float exterior = 1.0 - interior;

      vec2 textureCoordsL = textureCoords + vec2(-step_x, 0.0);
      vec2 textureCoordsLL = textureCoordsL + vec2(-step_x, 0.0);
      vec2 textureCoordsR = textureCoords + vec2(+step_x, 0.0);
      vec2 textureCoordsRR = textureCoordsR + vec2(+step_x, 0.0);
      vec2 textureCoordsT = textureCoords + vec2(0.0, +step_y);
      vec2 textureCoordsTT = textureCoordsT + vec2(0.0, +step_y);
      vec2 textureCoordsB = textureCoords + vec2(0.0, -step_y);
      vec2 textureCoordsBB = textureCoordsB + vec2(0.0, -step_y);
      
      vec4 RHS;
      vec4 RHS1;
      vec4 RHS2;
      vec4 updated;
      vec4 uvwq;
      vec4 uvwqL;
      vec4 uvwqLL;
      vec4 uvwqR;
      vec4 uvwqRR;
      vec4 uvwqT;
      vec4 uvwqTT;
      vec4 uvwqB;
      vec4 uvwqBB;
      vec4 uvwqLast;
  ` + update[type]
  );
}

/**
 * Generates shader code for ADER-DG timestepping.
 * @param {string} type - The DG stage identifier.
 * @returns {string} - The generated shader code.
 */
export function RDShaderMainDG(type) {
  let update = {};
  update.DGFE = `uvwq = dgSample2D(textureSource, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource, ix, iyBB, step_x, step_y, texWidth, texHeight);
    if (useFv) {
      uvwq = elemAvg;
      uvwqL = leftAvg;
      uvwqR = rightAvg;
      uvwqLL = left2Avg;
      uvwqRR = right2Avg;
      // Keep y-neighbors as sampled (don't zero out for 2D)
    }
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = dt * RHS / timescales + uvwq;`;
  // SSPRK/RK4 generic stage (Shu-Osher form):
  //   u^(i) = stageAlpha * u^n + stageDelta * u^(i-1) + stageBeta * dt * F(u^(i-1))
  // textureSource = u^n, textureSource1 = u^(i-1) (state for RHS evaluation)
  update.DGStage = `vec4 uvwq_orig = dgSample2D(textureSource, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwq = dgSample2D(textureSource1, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource1, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource1, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource1, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource1, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource1, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource1, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource1, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource1, ix, iyBB, step_x, step_y, texWidth, texHeight);
    if (useFv) {
      uvwq_orig = elemAvg;
      uvwq = elemAvg;
      uvwqL = leftAvg;
      uvwqR = rightAvg;
      uvwqLL = left2Avg;
      uvwqRR = right2Avg;
    }
    computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = stageAlpha * uvwq_orig + stageDelta * uvwq + stageBeta * dt * RHS / timescales;`;
  // RK4 final corrector stage:
  //   u^{n+1} = (-u^n + y2 + 2*y3 + y4)/3 + (dt/6)*F(y4)
  // textureSource = u^n, textureSource1 = y4, textureSource2 = y2, textureSource3 = y3
  update.DGRK4Corr = `vec4 uvwq_orig = dgSample2D(textureSource, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwq = dgSample2D(textureSource1, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource1, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource1, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource1, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource1, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource1, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource1, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource1, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource1, ix, iyBB, step_x, step_y, texWidth, texHeight);
    vec4 y2 = dgSample2D(textureSource2, ix, iy, step_x, step_y, texWidth, texHeight);
    vec4 y3 = dgSample2D(textureSource3, ix, iy, step_x, step_y, texWidth, texHeight);
    if (useFv) {
      uvwq_orig = elemAvg;
      uvwq = elemAvg;
      uvwqL = leftAvg;
      uvwqR = rightAvg;
      uvwqLL = left2Avg;
      uvwqRR = right2Avg;
    }
    computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = (-uvwq_orig + y2 + 2.0 * y3 + uvwq) / 3.0 + (dt / 6.0) * RHS / timescales;`;
  // ────────────────────────────────────────────────────────────────────
  // True ADER-DG shader types
  // ────────────────────────────────────────────────────────────────────

  // DGADERPred: Element-local predictor (zero-gradient at element boundaries).
  // This is the key ADER innovation: the predictor uses ONLY intra-element
  // spatial derivatives, removing inter-element numerical flux.
  // Computes: updated = stageAlpha * u^n + stageBeta * dt * F_local(src1)
  // textureSource = u^n, textureSource1 = state for element-local RHS evaluation
  update.DGADERPred = `vec4 uvwq_orig = dgSample2D(textureSource, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwq = dgSample2D(textureSource1, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource1, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource1, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource1, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource1, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource1, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource1, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource1, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource1, ix, iyBB, step_x, step_y, texWidth, texHeight);
    // Element-local: zero-gradient at element boundaries (no inter-element flux)
    // This removes the numerical flux contribution, making the predictor purely local.
    int localNode = ix - elem * nodesPerElem;
    if (localNode == 0) {
      // Left boundary of element: replace inter-element neighbors with boundary value
      uvwqL = uvwq;
      uvwqLL = uvwq;
    }
    if (localNode == order) {
      // Right boundary of element: replace inter-element neighbors with boundary value
      uvwqR = uvwq;
      uvwqRR = uvwq;
    }
    if (useFv) {
      uvwq_orig = elemAvg;
      uvwq = elemAvg;
      uvwqL = leftAvg;
      uvwqR = rightAvg;
      uvwqLL = left2Avg;
      uvwqRR = right2Avg;
    }
    computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = stageAlpha * uvwq_orig + stageBeta * dt * RHS / timescales;`;

  // DGADERPicard2: Dual-source element-local predictor for Picard iteration.
  // Evaluates F_local at two different predictor states (at two Gauss-Legendre
  // time points) and combines them with Picard integration weights.
  // Computes: updated = stageAlpha * u^n + stageWeight1*dt*F_local(src1) + stageWeight2*dt*F_local(src2)
  // textureSource = u^n, textureSource1 = q(tau_1), textureSource2 = q(tau_2)
  update.DGADERPicard2 = `vec4 uvwq_orig = dgSample2D(textureSource, ix, iy, step_x, step_y, texWidth, texHeight);
    int localNode = ix - elem * nodesPerElem;
    // --- Evaluate F_local at first temporal quadrature point (textureSource1) ---
    uvwq = dgSample2D(textureSource1, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource1, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource1, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource1, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource1, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource1, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource1, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource1, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource1, ix, iyBB, step_x, step_y, texWidth, texHeight);
    // Element-local: zero-gradient at element boundaries
    if (localNode == 0) { uvwqL = uvwq; uvwqLL = uvwq; }
    if (localNode == order) { uvwqR = uvwq; uvwqRR = uvwq; }
    if (useFv) {
      uvwq = elemAvg; uvwqL = leftAvg; uvwqR = rightAvg;
      uvwqLL = left2Avg; uvwqRR = right2Avg;
    }
    computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 RHS_tau1 = RHS;
    // --- Evaluate F_local at second temporal quadrature point (textureSource2) ---
    uvwq = dgSample2D(textureSource2, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource2, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource2, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource2, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource2, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource2, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource2, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource2, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource2, ix, iyBB, step_x, step_y, texWidth, texHeight);
    // Element-local: zero-gradient at element boundaries
    if (localNode == 0) { uvwqL = uvwq; uvwqLL = uvwq; }
    if (localNode == order) { uvwqR = uvwq; uvwqRR = uvwq; }
    if (useFv) {
      uvwq = elemAvg; uvwqL = leftAvg; uvwqR = rightAvg;
      uvwqLL = left2Avg; uvwqRR = right2Avg;
    }
    computeRHS(textureSource2, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    // Combine with Picard integration weights
    vec4 timescales = TIMESCALES;
    updated = stageAlpha * uvwq_orig + (stageWeight1 * RHS_tau1 + stageWeight2 * RHS) * dt / timescales;`;

  // DGADERCorr2: 2-point Gauss-Legendre corrector with full inter-element flux.
  // Evaluates F (with full DG flux) at two predictor states and combines
  // with equal GL weights (w1 = w2 = 1/2).
  // Computes: updated = u^n + dt/2 * [F(src1) + F(src2)]
  // textureSource = u^n, textureSource1 = q(tau_1), textureSource2 = q(tau_2)
  update.DGADERCorr2 = `vec4 uvwq_orig = dgSample2D(textureSource, ix, iy, step_x, step_y, texWidth, texHeight);
    // --- Evaluate F at first GL point (textureSource1) with full flux ---
    uvwq = dgSample2D(textureSource1, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource1, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource1, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource1, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource1, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource1, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource1, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource1, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource1, ix, iyBB, step_x, step_y, texWidth, texHeight);
    if (useFv) {
      uvwq = elemAvg; uvwqL = leftAvg; uvwqR = rightAvg;
      uvwqLL = left2Avg; uvwqRR = right2Avg;
    }
    computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 RHS_tau1 = RHS;
    // --- Evaluate F at second GL point (textureSource2) with full flux ---
    uvwq = dgSample2D(textureSource2, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource2, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource2, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource2, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource2, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource2, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource2, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource2, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource2, ix, iyBB, step_x, step_y, texWidth, texHeight);
    if (useFv) {
      uvwq = elemAvg; uvwqL = leftAvg; uvwqR = rightAvg;
      uvwqLL = left2Avg; uvwqRR = right2Avg;
    }
    computeRHS(textureSource2, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    // 2-point Gauss-Legendre: w1 = w2 = 1/2
    vec4 timescales = TIMESCALES;
    updated = uvwq_orig + 0.5 * dt * (RHS_tau1 + RHS) / timescales;`;

  // ────────────────────────────────────────────────────────────────────
  // Fused corrector + boundary averaging shaders (Fusion A)
  // At boundary nodes, recompute the corrector at the neighbor position
  // and average the two results, eliminating the separate BoundaryAvg pass.
  // ────────────────────────────────────────────────────────────────────

  // Helper: generates GLSL to compute corrector 'updated' for an arbitrary ix position.
  // srcSampler is a GLSL expression for the sampler (e.g., "textureSource1").
  // 'varPrefix' avoids name collisions when computing neighbor values.
  const fusedBoundaryAvgBlock = (correctorBody) => `
    // --- Fused boundary averaging ---
    int localNode = ix - elem * nodesPerElem;
    if (localNode == order || localNode == 0) {
      // We are at a shared boundary node. Compute the corrector at the neighbor's position too.
      vec4 myUpdated = updated;
      int neighborIx;
      if (localNode == order) {
        int nextElem = (elem + 1) % nElem;
        neighborIx = nextElem * nodesPerElem;
      } else {
        int prevElem = (elem - 1 + nElem) % nElem;
        neighborIx = prevElem * nodesPerElem + order;
      }
      // Recompute stencil indices for the neighbor position
      int nbIxL = dgStepLeft(neighborIx, order, texWidth);
      int nbIxR = dgStepRight(neighborIx, order, texWidth);
      int nbIxLL = dgStepLeft(nbIxL, order, texWidth);
      int nbIxRR = dgStepRight(nbIxR, order, texWidth);
      // Recompute corrector at neighbor position
      ${correctorBody}
      // Average own result with neighbor's result
      updated = 0.5 * (myUpdated + updated);
    }`;

  // Fused Forward Euler + boundary avg (Order 1)
  update.DGFEFused = `uvwq = dgSample2D(textureSource, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource, ix, iyBB, step_x, step_y, texWidth, texHeight);
    if (useFv) {
      uvwq = elemAvg;
      uvwqL = leftAvg;
      uvwqR = rightAvg;
      uvwqLL = left2Avg;
      uvwqRR = right2Avg;
    }
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = dt * RHS / timescales + uvwq;
    ` + fusedBoundaryAvgBlock(`
      uvwq = dgSample2D(textureSource, neighborIx, iy, step_x, step_y, texWidth, texHeight);
      uvwqL = dgSample2D(textureSource, nbIxL, iy, step_x, step_y, texWidth, texHeight);
      uvwqR = dgSample2D(textureSource, nbIxR, iy, step_x, step_y, texWidth, texHeight);
      uvwqLL = dgSample2D(textureSource, nbIxLL, iy, step_x, step_y, texWidth, texHeight);
      uvwqRR = dgSample2D(textureSource, nbIxRR, iy, step_x, step_y, texWidth, texHeight);
      uvwqT = dgSample2D(textureSource, neighborIx, iyT, step_x, step_y, texWidth, texHeight);
      uvwqB = dgSample2D(textureSource, neighborIx, iyB, step_x, step_y, texWidth, texHeight);
      uvwqTT = dgSample2D(textureSource, neighborIx, iyTT, step_x, step_y, texWidth, texHeight);
      uvwqBB = dgSample2D(textureSource, neighborIx, iyBB, step_x, step_y, texWidth, texHeight);
      computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
      updated = dt * RHS / timescales + uvwq;
    `);

  // Fused DGStage corrector + boundary avg (Order 2)
  update.DGStageFused = `vec4 uvwq_orig = dgSample2D(textureSource, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwq = dgSample2D(textureSource1, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource1, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource1, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource1, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource1, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource1, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource1, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource1, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource1, ix, iyBB, step_x, step_y, texWidth, texHeight);
    if (useFv) {
      uvwq_orig = elemAvg;
      uvwq = elemAvg;
      uvwqL = leftAvg;
      uvwqR = rightAvg;
      uvwqLL = left2Avg;
      uvwqRR = right2Avg;
    }
    computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = stageAlpha * uvwq_orig + stageDelta * uvwq + stageBeta * dt * RHS / timescales;
    ` + fusedBoundaryAvgBlock(`
      vec4 nbOrig = dgSample2D(textureSource, neighborIx, iy, step_x, step_y, texWidth, texHeight);
      uvwq = dgSample2D(textureSource1, neighborIx, iy, step_x, step_y, texWidth, texHeight);
      uvwqL = dgSample2D(textureSource1, nbIxL, iy, step_x, step_y, texWidth, texHeight);
      uvwqR = dgSample2D(textureSource1, nbIxR, iy, step_x, step_y, texWidth, texHeight);
      uvwqLL = dgSample2D(textureSource1, nbIxLL, iy, step_x, step_y, texWidth, texHeight);
      uvwqRR = dgSample2D(textureSource1, nbIxRR, iy, step_x, step_y, texWidth, texHeight);
      uvwqT = dgSample2D(textureSource1, neighborIx, iyT, step_x, step_y, texWidth, texHeight);
      uvwqB = dgSample2D(textureSource1, neighborIx, iyB, step_x, step_y, texWidth, texHeight);
      uvwqTT = dgSample2D(textureSource1, neighborIx, iyTT, step_x, step_y, texWidth, texHeight);
      uvwqBB = dgSample2D(textureSource1, neighborIx, iyBB, step_x, step_y, texWidth, texHeight);
      computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
      updated = stageAlpha * nbOrig + stageDelta * uvwq + stageBeta * dt * RHS / timescales;
    `);

  // Fused DGADERCorr2 corrector + boundary avg (Orders 3-4)
  update.DGADERCorr2Fused = `vec4 uvwq_orig = dgSample2D(textureSource, ix, iy, step_x, step_y, texWidth, texHeight);
    // --- Evaluate F at first GL point (textureSource1) with full flux ---
    uvwq = dgSample2D(textureSource1, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource1, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource1, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource1, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource1, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource1, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource1, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource1, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource1, ix, iyBB, step_x, step_y, texWidth, texHeight);
    if (useFv) {
      uvwq = elemAvg; uvwqL = leftAvg; uvwqR = rightAvg;
      uvwqLL = left2Avg; uvwqRR = right2Avg;
    }
    computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 RHS_tau1 = RHS;
    // --- Evaluate F at second GL point (textureSource2) with full flux ---
    uvwq = dgSample2D(textureSource2, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource2, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource2, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource2, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource2, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource2, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource2, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource2, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource2, ix, iyBB, step_x, step_y, texWidth, texHeight);
    if (useFv) {
      uvwq = elemAvg; uvwqL = leftAvg; uvwqR = rightAvg;
      uvwqLL = left2Avg; uvwqRR = right2Avg;
    }
    computeRHS(textureSource2, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = uvwq_orig + 0.5 * dt * (RHS_tau1 + RHS) / timescales;
    ` + fusedBoundaryAvgBlock(`
      vec4 nbOrig = dgSample2D(textureSource, neighborIx, iy, step_x, step_y, texWidth, texHeight);
      // F at first GL point for neighbor
      uvwq = dgSample2D(textureSource1, neighborIx, iy, step_x, step_y, texWidth, texHeight);
      uvwqL = dgSample2D(textureSource1, nbIxL, iy, step_x, step_y, texWidth, texHeight);
      uvwqR = dgSample2D(textureSource1, nbIxR, iy, step_x, step_y, texWidth, texHeight);
      uvwqLL = dgSample2D(textureSource1, nbIxLL, iy, step_x, step_y, texWidth, texHeight);
      uvwqRR = dgSample2D(textureSource1, nbIxRR, iy, step_x, step_y, texWidth, texHeight);
      uvwqT = dgSample2D(textureSource1, neighborIx, iyT, step_x, step_y, texWidth, texHeight);
      uvwqB = dgSample2D(textureSource1, neighborIx, iyB, step_x, step_y, texWidth, texHeight);
      uvwqTT = dgSample2D(textureSource1, neighborIx, iyTT, step_x, step_y, texWidth, texHeight);
      uvwqBB = dgSample2D(textureSource1, neighborIx, iyBB, step_x, step_y, texWidth, texHeight);
      computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
      vec4 nbRHS1 = RHS;
      // F at second GL point for neighbor
      uvwq = dgSample2D(textureSource2, neighborIx, iy, step_x, step_y, texWidth, texHeight);
      uvwqL = dgSample2D(textureSource2, nbIxL, iy, step_x, step_y, texWidth, texHeight);
      uvwqR = dgSample2D(textureSource2, nbIxR, iy, step_x, step_y, texWidth, texHeight);
      uvwqLL = dgSample2D(textureSource2, nbIxLL, iy, step_x, step_y, texWidth, texHeight);
      uvwqRR = dgSample2D(textureSource2, nbIxRR, iy, step_x, step_y, texWidth, texHeight);
      uvwqT = dgSample2D(textureSource2, neighborIx, iyT, step_x, step_y, texWidth, texHeight);
      uvwqB = dgSample2D(textureSource2, neighborIx, iyB, step_x, step_y, texWidth, texHeight);
      uvwqTT = dgSample2D(textureSource2, neighborIx, iyTT, step_x, step_y, texWidth, texHeight);
      uvwqBB = dgSample2D(textureSource2, neighborIx, iyBB, step_x, step_y, texWidth, texHeight);
      computeRHS(textureSource2, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
      updated = nbOrig + 0.5 * dt * (nbRHS1 + RHS) / timescales;
    `);

  // ────────────────────────────────────────────────────────────────────
  // MRT (Multiple Render Target) shader types (Fusion B & C)
  // These evaluate computeRHS once and write two outputs with different
  // coefficients, using gl_FragData[0] and gl_FragData[1].
  // ────────────────────────────────────────────────────────────────────

  // DGADERPredMRT: Element-local predictor writing two outputs with different beta values.
  // gl_FragColor = u^n + stageBeta  * dt * F_local(src1)   [location 0]
  // fragData1    = u^n + stageBeta2 * dt * F_local(src1)   [location 1]
  update.DGADERPredMRT = `vec4 uvwq_orig = dgSample2D(textureSource, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwq = dgSample2D(textureSource1, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource1, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource1, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource1, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource1, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource1, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource1, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource1, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource1, ix, iyBB, step_x, step_y, texWidth, texHeight);
    int localNode = ix - elem * nodesPerElem;
    if (localNode == 0) { uvwqL = uvwq; uvwqLL = uvwq; }
    if (localNode == order) { uvwqR = uvwq; uvwqRR = uvwq; }
    if (useFv) {
      uvwq_orig = elemAvg; uvwq = elemAvg; uvwqL = leftAvg; uvwqR = rightAvg;
      uvwqLL = left2Avg; uvwqRR = right2Avg;
    }
    computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    vec4 scaledRHS = dt * RHS / timescales;
    gl_FragColor = stageAlpha * uvwq_orig + stageBeta * scaledRHS;
    fragData1 = stageAlpha * uvwq_orig + stageBeta2 * scaledRHS;`;

  // DGADERPicard2MRT: Dual-source element-local Picard writing two outputs with different weight pairs.
  // gl_FragColor = u^n + (stageWeight1  * F(src1) + stageWeight2  * F(src2)) * dt  [location 0]
  // fragData1    = u^n + (stageWeight1b * F(src1) + stageWeight2b * F(src2)) * dt  [location 1]
  update.DGADERPicard2MRT = `vec4 uvwq_orig = dgSample2D(textureSource, ix, iy, step_x, step_y, texWidth, texHeight);
    int localNode = ix - elem * nodesPerElem;
    // --- F_local at first temporal quadrature point (textureSource1) ---
    uvwq = dgSample2D(textureSource1, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource1, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource1, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource1, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource1, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource1, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource1, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource1, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource1, ix, iyBB, step_x, step_y, texWidth, texHeight);
    if (localNode == 0) { uvwqL = uvwq; uvwqLL = uvwq; }
    if (localNode == order) { uvwqR = uvwq; uvwqRR = uvwq; }
    if (useFv) {
      uvwq = elemAvg; uvwqL = leftAvg; uvwqR = rightAvg;
      uvwqLL = left2Avg; uvwqRR = right2Avg;
    }
    computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 RHS_tau1 = RHS;
    // --- F_local at second temporal quadrature point (textureSource2) ---
    uvwq = dgSample2D(textureSource2, ix, iy, step_x, step_y, texWidth, texHeight);
    uvwqL = dgSample2D(textureSource2, ixL, iy, step_x, step_y, texWidth, texHeight);
    uvwqR = dgSample2D(textureSource2, ixR, iy, step_x, step_y, texWidth, texHeight);
    uvwqLL = dgSample2D(textureSource2, ixLL, iy, step_x, step_y, texWidth, texHeight);
    uvwqRR = dgSample2D(textureSource2, ixRR, iy, step_x, step_y, texWidth, texHeight);
    uvwqT = dgSample2D(textureSource2, ix, iyT, step_x, step_y, texWidth, texHeight);
    uvwqB = dgSample2D(textureSource2, ix, iyB, step_x, step_y, texWidth, texHeight);
    uvwqTT = dgSample2D(textureSource2, ix, iyTT, step_x, step_y, texWidth, texHeight);
    uvwqBB = dgSample2D(textureSource2, ix, iyBB, step_x, step_y, texWidth, texHeight);
    if (localNode == 0) { uvwqL = uvwq; uvwqLL = uvwq; }
    if (localNode == order) { uvwqR = uvwq; uvwqRR = uvwq; }
    if (useFv) {
      uvwq = elemAvg; uvwqL = leftAvg; uvwqR = rightAvg;
      uvwqLL = left2Avg; uvwqRR = right2Avg;
    }
    computeRHS(textureSource2, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    gl_FragColor = stageAlpha * uvwq_orig + (stageWeight1  * RHS_tau1 + stageWeight2  * RHS) * dt / timescales;
    fragData1 = stageAlpha * uvwq_orig + (stageWeight1b * RHS_tau1 + stageWeight2b * RHS) * dt / timescales;`;

  return (
    `
  void main()
  {
      ivec2 texSize = textureSize(textureSource,0);
      float step_x = 1.0 / float(texSize.x);
      float step_y = 1.0 / float(texSize.y);
      int order = int(dgOrder + 0.5);
      int nodesPerElem = order + 1;
      int texWidth = texSize.x;
      int texHeight = texSize.y;
      int ix = int(floor(textureCoords.x * float(texWidth)));
      int iy = int(floor(textureCoords.y * float(texHeight)));
      int ixL = dgStepLeft(ix, order, texWidth);
      int ixR = dgStepRight(ix, order, texWidth);
      int ixLL = dgStepLeft(ixL, order, texWidth);
      int ixRR = dgStepRight(ixR, order, texWidth);
      // Y-direction uses simple stepping (regular grid, not DG structure)
      int iyB = iy - 1;
      if (iyB < 0) iyB += texHeight;
      int iyT = iy + 1;
      if (iyT >= texHeight) iyT -= texHeight;
      int iyBB = iyB - 1;
      if (iyBB < 0) iyBB += texHeight;
      int iyTT = iyT + 1;
      if (iyTT >= texHeight) iyTT -= texHeight;
      int elem = ix / nodesPerElem;
      int nElem = max(1, texWidth / nodesPerElem);
      int elemL = elem - 1;
      if (elemL < 0) {
        elemL += nElem;
      }
      int elemR = elem + 1;
      if (elemR >= nElem) {
        elemR -= nElem;
      }
      int elemLL = elemL - 1;
      if (elemLL < 0) {
        elemLL += nElem;
      }
      int elemRR = elemR + 1;
      if (elemRR >= nElem) {
        elemRR -= nElem;
      }

      vec4 elemMin;
      vec4 elemMax;
      vec4 leftMin;
      vec4 leftMax;
      vec4 rightMin;
      vec4 rightMax;
      dgElementMinMax(textureSource, elem, order, nodesPerElem, texWidth, step_x, elemMin, elemMax);
      dgElementMinMax(textureSource, elemL, order, nodesPerElem, texWidth, step_x, leftMin, leftMax);
      dgElementMinMax(textureSource, elemR, order, nodesPerElem, texWidth, step_x, rightMin, rightMax);
      vec4 neighborMin = min(leftMin, rightMin);
      vec4 neighborMax = max(leftMax, rightMax);
      vec4 range = neighborMax - neighborMin;
      vec4 uvwqCheck = dgSample2D(textureSource, ix, iy, step_x, step_y, texWidth, texHeight);
      vec4 lower = neighborMin - 0.5 * range;
      vec4 upper = neighborMax + 0.5 * range;
      bvec4 lowMask = lessThan(uvwqCheck, lower);
      bvec4 highMask = greaterThan(uvwqCheck, upper);
      bvec4 rangeMask = greaterThan(range, vec4(1.0e-6));
      // Disable TVD limiter for 2D (texHeight > 1) as it causes blocky artifacts
      bool is2D = texHeight > 1;
      bool useFv = !is2D && (any(lowMask) || any(highMask)) && any(rangeMask);
      vec4 elemAvg = dgElementAverage(textureSource, elem, order, nodesPerElem, texWidth, step_x);
      vec4 leftAvg = dgElementAverage(textureSource, elemL, order, nodesPerElem, texWidth, step_x);
      vec4 rightAvg = dgElementAverage(textureSource, elemR, order, nodesPerElem, texWidth, step_x);
      vec4 left2Avg = dgElementAverage(textureSource, elemLL, order, nodesPerElem, texWidth, step_x);
      vec4 right2Avg = dgElementAverage(textureSource, elemRR, order, nodesPerElem, texWidth, step_x);

      vec4 RHS;
      vec4 RHS1;
      vec4 RHS2;
      vec4 updated;
      vec4 uvwq;
      vec4 uvwqL;
      vec4 uvwqLL;
      vec4 uvwqR;
      vec4 uvwqRR;
      vec4 uvwqT;
      vec4 uvwqTT;
      vec4 uvwqB;
      vec4 uvwqBB;
      vec4 uvwqLast;
      ` + update[type]
  );
}

/**
 * Returns the shader code for a reaction-diffusion simulation with periodic boundary conditions.
 * @returns {string} The shader code.
 */
export function RDShaderPeriodic() {
  return ``;
}

/**
 * Generates shader code for specifying the values of ghost cells in the x-direction.
 * @param {string} [LR] - Determines whether to apply the condition at the left ("L"), right ("R"), or both ("LR"). If undefined, returns both.
 * @returns {string} The shader code for setting the species of ghost cells in the x-direction.
 */
export function RDShaderGhostX(LR) {
  const L = `
    if (textureCoords.x - step_x < 0.0) {
        uvwqL.SPECIES = GHOSTSPECIES;
    }
    `;
  const R = `
    if (textureCoords.x + step_x > 1.0) {
        uvwqR.SPECIES = GHOSTSPECIES;
    }
    `;
  if (LR == undefined) return L + R;
  if (LR == "L") return L;
  if (LR == "R") return R;
  return "";
}

/**
 * Generates shader code for specifying the values of ghost cells in the y-direction.
 * @param {string} [TB] - Determines whether to apply the condition at the top ("T"), bottom ("B"), or both ("TB"). If undefined, returns both.
 * @returns {string} The shader code for setting the species of ghost cells in the y-direction.
 */
export function RDShaderGhostY(TB) {
  const T = `
    if (textureCoords.y + step_y > 1.0){
        uvwqT.SPECIES = GHOSTSPECIES;
    }
    `;
  const B = `
    if (textureCoords.y - step_y < 0.0) {
        uvwqB.SPECIES = GHOSTSPECIES;
    }
    `;
  if (TB == undefined) return T + B;
  if (TB == "T") return T;
  if (TB == "B") return B;
  return "";
}

/**
 * Returns a string containing the Robin boundary condition shader code in the x-direction.
 * @param {string} [LR] - Determines whether to apply the condition at the left ("L"), right ("R"), or both ("LR"). If undefined, returns both.
 * @returns {string} The Robin boundary condition shader code.
 */
export function RDShaderRobinX(LR) {
  const L = `
    if (textureCoords.x - step_x < 0.0) {
        uvwqL.SPECIES = 2.0 * (dx * robinRHSSPECIESL) + uvwqR.SPECIES;
    }
    `;
  const R = `
    if (textureCoords.x + step_x > 1.0) {
        uvwqR.SPECIES = 2.0 * (dx * robinRHSSPECIESR) + uvwqL.SPECIES;
    }
    `;
  if (LR == undefined) return L + R;
  if (LR == "L") return L;
  if (LR == "R") return R;
  return "";
}

/**
 * Returns a string containing the Robin boundary condition shader code in the y-direction.
 * @param {string} [TB] - Determines whether to apply the condition at the top ("T"), bottom ("B"), or both ("TB"). If undefined, returns both.
 * @returns {string} The Robin boundary condition shader code.
 */
export function RDShaderRobinY(TB) {
  const T = `
    if (textureCoords.y + step_y > 1.0){
        uvwqT.SPECIES = 2.0 * (dy * robinRHSSPECIEST) + uvwqB.SPECIES;
    }
    `;
  const B = `
    if (textureCoords.y - step_y < 0.0) {
        uvwqB.SPECIES = 2.0 * (dy * robinRHSSPECIESB) + uvwqT.SPECIES;
    }
    `;
  if (TB == undefined) return T + B;
  if (TB == "T") return T;
  if (TB == "B") return B;
  return "";
}

/**
 * Generates a Robin boundary condition shader for a custom domain in the x-direction.
 * @param {string} TB - Determines whether to apply the condition at the left ("L"), right ("R"), or both ("LR"). If undefined, returns both.
 * @param {string} fun - A function that defines the custom domain.
 * @returns {string} The generated shader code.
 */
export function RDShaderRobinCustomDomainX(LR, fun) {
  const L = `
    if (float(indicatorFunL) <= 0.0 || textureCoords.x - 2.0*step_x < 0.0) {
      if (float(indicatorFunR) <= 0.0) {
        uvwqL.SPECIES = dx * robinRHSSPECIESL + uvwq.SPECIES;
      } else {
        uvwqL.SPECIES = 2.0 * (dx * robinRHSSPECIESL) + uvwqR.SPECIES;
      }
    }
    `
    .replace(
      /indicatorFunL/,
      fun.replaceAll(/\bx\b/g, "(x-1.25*dx)").replaceAll(/\buvwq\./g, "uvwqL."),
    )
    .replace(
      /indicatorFunR/,
      fun.replaceAll(/\bx\b/g, "(x+1.25*dx)").replaceAll(/\buvwq\./g, "uvwqR."),
    );
  const R = `
    if (float(indicatorFunR) <= 0.0 || textureCoords.x + 2.0*step_x > 1.0) {
      if (float(indicatorFunL) <= 0.0) {
        uvwqR.SPECIES = dx * robinRHSSPECIESR + uvwq.SPECIES;
      } else {
        uvwqR.SPECIES = 2.0 * (dx * robinRHSSPECIESR) + uvwqL.SPECIES;
      }
    }
    `
    .replace(
      /indicatorFunR/,
      fun.replaceAll(/\bx\b/g, "(x+1.25*dx)").replaceAll(/\buvwq\./g, "uvwqR."),
    )
    .replace(
      /indicatorFunL/,
      fun.replaceAll(/\bx\b/g, "(x-1.25*dx)").replaceAll(/\buvwq\./g, "uvwqL."),
    );
  if (LR == undefined) return L + R;
  if (LR == "L") return L;
  if (LR == "R") return R;
  return "";
}

/**
 * Generates a Robin boundary condition shader for a custom domain in the y-direction.
 * @param {string} TB - Determines whether to apply the condition at the top ("T"), bottom ("B"), or both ("TB"). If undefined, returns both.
 * @param {string} fun - A function that defines the custom domain.
 * @returns {string} The generated shader code.
 */
export function RDShaderRobinCustomDomainY(TB, fun) {
  const T = `
    if (float(indicatorFunT) <= 0.0 || textureCoords.y + 2.0*step_y > 1.0){
      if (float(indicatorFunB) <= 0.0) {
        uvwqT.SPECIES = dy * robinRHSSPECIEST + uvwq.SPECIES;
      } else {
        uvwqT.SPECIES = 2.0 * (dy * robinRHSSPECIEST) + uvwqB.SPECIES;
      }
    }
    `
    .replace(
      /indicatorFunT/,
      fun.replaceAll(/\by\b/g, "(y+1.25*dy)").replaceAll(/\buvwq\./g, "uvwqT."),
    )
    .replace(
      /indicatorFunB/,
      fun.replaceAll(/\by\b/g, "(y-1.25*dy)").replaceAll(/\buvwq\./g, "uvwqB."),
    );
  const B = `
    if (float(indicatorFunB) <= 0.0 || textureCoords.y - 2.0*step_y < 0.0) {
      if (float(indicatorFunT) <= 0.0) {
        uvwqB.SPECIES = dy * robinRHSSPECIESB + uvwq.SPECIES;
      } else {
        uvwqB.SPECIES = 2.0 * (dy * robinRHSSPECIESB) + uvwqT.SPECIES;
      }
    }
    `
    .replace(
      /indicatorFunB/,
      fun.replaceAll(/\by\b/g, "(y-1.25*dy)").replaceAll(/\buvwq\./g, "uvwqB."),
    )
    .replace(
      /indicatorFunT/,
      fun.replaceAll(/\by\b/g, "(y+1.25*dy)").replaceAll(/\buvwq\./g, "uvwqT."),
    );
  if (TB == undefined) return T + B;
  if (TB == "T") return T;
  if (TB == "B") return B;
  return "";
}

/**
 * Returns the shader code for computing advection before boundary conditions have been applied.
 * @returns {string} The shader code.
 */
export function RDShaderAdvectionPreBC() {
  return `
    vec4 uvwqX = (uvwqR - uvwqL) / (2.0*dx);
    vec4 uvwqY = (uvwqT - uvwqB) / (2.0*dy);
    vec4 uvwqXF = (uvwqR - uvwq) / dx;
    vec4 uvwqYF = (uvwqT - uvwq) / dy;
    vec4 uvwqXB = (uvwq - uvwqL) / dx;
    vec4 uvwqYB = (uvwq - uvwqB) / dy;
    vec4 uvwqXFXF = (4.0*uvwqR - 3.0*uvwq - uvwqRR) / (2.0*dx);
    vec4 uvwqYFYF = (4.0*uvwqT - 3.0*uvwq - uvwqTT) / (2.0*dy);
    vec4 uvwqXBXB = (3.0*uvwq - 4.0*uvwqL + uvwqLL) / (2.0*dx);
    vec4 uvwqYBYB = (3.0*uvwq - 4.0*uvwqB + uvwqBB) / (2.0*dy);
    `;
}

/**
 * Returns the shader code for computing advection after boundary conditions have been applied.
 * @returns {string} The shader code.
 */
export function RDShaderAdvectionPostBC() {
  return `
    uvwqX = (uvwqR - uvwqL) / (2.0*dx);
    uvwqY = (uvwqT - uvwqB) / (2.0*dy);
    uvwqXF = (uvwqR - uvwq) / dx;
    uvwqYF = (uvwqT - uvwq) / dy;
    uvwqXB = (uvwq - uvwqL) / dx;
    uvwqYB = (uvwq - uvwqB) / dy;
    uvwqXFXF = (4.0*uvwqR - 3.0*uvwq - uvwqRR) / (2.0*dx);
    uvwqYFYF = (4.0*uvwqT - 3.0*uvwq - uvwqTT) / (2.0*dy);
    uvwqXBXB = (3.0*uvwq - 4.0*uvwqL + uvwqLL) / (2.0*dx);
    uvwqYBYB = (3.0*uvwq - 4.0*uvwqB + uvwqBB) / (2.0*dy);
    `;
}

/**
 * Returns the shader code for computing diffusion before boundary conditions have been applied.
 * @returns {string} The shader code.
 */
export function RDShaderDiffusionPreBC() {
  return `
    vec4 uvwqXX = (uvwqR - 2.0*uvwq + uvwqL) / (dx*dx);
    vec4 uvwqYY = (uvwqT - 2.0*uvwq + uvwqB) / (dy*dy);
    `;
}

/**
 * Returns the shader code for computing diffusion after boundary conditions have been applied.
 * @returns {string} The shader code.
 */
export function RDShaderDiffusionPostBC() {
  return `
    uvwqXX = (uvwqR - 2.0*uvwq + uvwqL) / (dx*dx);
    uvwqYY = (uvwqT - 2.0*uvwq + uvwqB) / (dy*dy);
    `;
}

/**
 * Generates a shader for updating a reaction-diffusion system without cross diffusion.
 * @param {number} [numSpecies=4] - The number of species. Defaults to 4.
 * @returns {string} - The shader code for the update.
 */
export function RDShaderUpdateNormal(numSpecies) {
  if (numSpecies == undefined) numSpecies = 4;
  let shader = "";
  shader += `
  float LDuuU = 0.5*((Duux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DuuxR*(uvwqR.r - uvwq.r) + DuuxL*(uvwqL.r - uvwq.r)) / dx) / dx +  0.5*((Duuy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DuuyT*(uvwqT.r - uvwq.r) + DuuyB*(uvwqB.r - uvwq.r)) / dy) / dy;
  float du = LDuuU + UFUN;
  `;
  if (numSpecies > 1) {
    shader += `
    float LDvvV = 0.5*((Dvvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DvvxR*(uvwqR.g - uvwq.g) + DvvxL*(uvwqL.g - uvwq.g)) / dx) / dx +  0.5*((Dvvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DvvyT*(uvwqT.g - uvwq.g) + DvvyB*(uvwqB.g - uvwq.g)) / dy) / dy;
    float dv = LDvvV + VFUN;
    `;
  }
  if (numSpecies > 2) {
    shader += `
    float LDwwW = 0.5*((Dwwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DwwxR*(uvwqR.b - uvwq.b) + DwwxL*(uvwqL.b - uvwq.b)) / dx) / dx +  0.5*((Dwwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DwwyT*(uvwqT.b - uvwq.b) + DwwyB*(uvwqB.b - uvwq.b)) / dy) / dy;
    float dw = LDwwW + WFUN;
    `;
  }
  if (numSpecies > 3) {
    shader += `
    float LDqqQ = 0.5*((Dqqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DqqxR*(uvwqR.a - uvwq.a) + DqqxL*(uvwqL.a - uvwq.a)) / dx) / dx +  0.5*((Dqqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DqqyT*(uvwqT.a - uvwq.a) + DqqyB*(uvwqB.a - uvwq.a)) / dy) / dy;
    float dq = LDqqQ + QFUN;
    `;
  }
  // Add the final line of the shader.
  switch (numSpecies) {
    case 1:
      shader += `result = vec4(du,0.0,0.0,0.0);`;
      break;
    case 2:
      shader += `result = vec4(du,dv,0.0,0.0);`;
      break;
    case 3:
      shader += `result = vec4(du,dv,dw,0.0);`;
      break;
    case 4:
      shader += `result = vec4(du,dv,dw,dq);`;
      break;
  }
  return (
    shader +
    `
    }`
  );
}

/**
 * Generates a shader for updating a reaction-diffusion system with cross diffusion.
 * @param {number} [numSpecies=4] - The number of species in the system.
 * @returns {string} The generated shader code.
 */
export function RDShaderUpdateCross(numSpecies) {
  if (numSpecies == undefined) numSpecies = 4;
  let shader = "";
  shader +=
    [
      `vec2 LDuuU = vec2(Duux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DuuxR*(uvwqR.r - uvwq.r) + DuuxL*(uvwqL.r - uvwq.r), Duuy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DuuyT*(uvwqT.r - uvwq.r) + DuuyB*(uvwqB.r - uvwq.r));`,
      `vec2 LDuvV = vec2(Duvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DuvxR*(uvwqR.g - uvwq.g) + DuvxL*(uvwqL.g - uvwq.g), Duvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DuvyT*(uvwqT.g - uvwq.g) + DuvyB*(uvwqB.g - uvwq.g));`,
      `vec2 LDuwW = vec2(Duwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DuwxR*(uvwqR.b - uvwq.b) + DuwxL*(uvwqL.b - uvwq.b), Duwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DuwyT*(uvwqT.b - uvwq.b) + DuwyB*(uvwqB.b - uvwq.b));`,
      `vec2 LDuqQ = vec2(Duqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DuqxR*(uvwqR.a - uvwq.a) + DuqxL*(uvwqL.a - uvwq.a), Duqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DuqyT*(uvwqT.a - uvwq.a) + DuqyB*(uvwqB.a - uvwq.a));`,
    ]
      .slice(0, numSpecies)
      .join("\n") +
    `\nfloat du = 0.5*dot(dSquared,` +
    [`LDuuU`, `LDuvV`, `LDuwW`, `LDuqQ`].slice(0, numSpecies).join(" + ") +
    `) + UFUN;\n`;
  // If there is more than one species, add the second species.
  if (numSpecies > 1) {
    // Compute the cross-diffusion terms.
    shader +=
      [
        `vec2 LDvuU = vec2(Dvux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DvuxR*(uvwqR.r - uvwq.r) + DvuxL*(uvwqL.r - uvwq.r), Dvuy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DvuyT*(uvwqT.r - uvwq.r) + DvuyB*(uvwqB.r - uvwq.r));`,
        `vec2 LDvvV = vec2(Dvvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DvvxR*(uvwqR.g - uvwq.g) + DvvxL*(uvwqL.g - uvwq.g), Dvvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DvvyT*(uvwqT.g - uvwq.g) + DvvyB*(uvwqB.g - uvwq.g));`,
        `vec2 LDvwW = vec2(Dvwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DvwxR*(uvwqR.b - uvwq.b) + DvwxL*(uvwqL.b - uvwq.b), Dvwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DvwyT*(uvwqT.b - uvwq.b) + DvwyB*(uvwqB.b - uvwq.b));`,
        `vec2 LDvqQ = vec2(Dvqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DvqxR*(uvwqR.a - uvwq.a) + DvqxL*(uvwqL.a - uvwq.a), Dvqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DvqyT*(uvwqT.a - uvwq.a) + DvqyB*(uvwqB.a - uvwq.a));`,
      ]
        .slice(0, numSpecies)
        .join("\n") +
      `\nfloat dv = 0.5*dot(dSquared,` +
      [`LDvuU`, `LDvvV`, `LDvwW`, `LDvqQ`].slice(0, numSpecies).join(" + ") +
      `) + VFUN;\n`;
  }
  // If there are more than two species, add the third species.
  if (numSpecies > 2) {
    // Compute the cross-diffusion terms.
    shader +=
      [
        `vec2 LDwuU = vec2(Dwux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DwuxR*(uvwqR.r - uvwq.r) + DwuxL*(uvwqL.r - uvwq.r), Dwuy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DwuyT*(uvwqT.r - uvwq.r) + DwuyB*(uvwqB.r - uvwq.r));`,
        `vec2 LDwvV = vec2(Dwvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DwvxR*(uvwqR.g - uvwq.g) + DwvxL*(uvwqL.g - uvwq.g), Dwvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DwvyT*(uvwqT.g - uvwq.g) + DwvyB*(uvwqB.g - uvwq.g));`,
        `vec2 LDwwW = vec2(Dwwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DwwxR*(uvwqR.b - uvwq.b) + DwwxL*(uvwqL.b - uvwq.b), Dwwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DwwyT*(uvwqT.b - uvwq.b) + DwwyB*(uvwqB.b - uvwq.b));`,
        `vec2 LDwqQ = vec2(Dwqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DwqxR*(uvwqR.a - uvwq.a) + DwqxL*(uvwqL.a - uvwq.a), Dwqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DwqyT*(uvwqT.a - uvwq.a) + DwqyB*(uvwqB.a - uvwq.a));`,
      ]
        .slice(0, numSpecies)
        .join("\n") +
      `\nfloat dw = 0.5*dot(dSquared,` +
      [`LDwuU`, `LDwvV`, `LDwwW`, `LDwqQ`].slice(0, numSpecies).join(" + ") +
      `) + WFUN;\n`;
  }
  // If there are more than three species, add the fourth species.
  if (numSpecies > 3) {
    // Compute the cross-diffusion terms.
    shader +=
      [
        `vec2 LDquU = vec2(Dqux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DquxR*(uvwqR.r - uvwq.r) + DquxL*(uvwqL.r - uvwq.r), Dquy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DquyT*(uvwqT.r - uvwq.r) + DquyB*(uvwqB.r - uvwq.r));`,
        `vec2 LDqvV = vec2(Dqvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DqvxR*(uvwqR.g - uvwq.g) + DqvxL*(uvwqL.g - uvwq.g), Dqvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DqvyT*(uvwqT.g - uvwq.g) + DqvyB*(uvwqB.g - uvwq.g));`,
        `vec2 LDqwW = vec2(Dqwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DqwxR*(uvwqR.b - uvwq.b) + DqwxL*(uvwqL.b - uvwq.b), Dqwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DqwyT*(uvwqT.b - uvwq.b) + DqwyB*(uvwqB.b - uvwq.b));`,
        `vec2 LDqqQ = vec2(Dqqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DqqxR*(uvwqR.a - uvwq.a) + DqqxL*(uvwqL.a - uvwq.a), Dqqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DqqyT*(uvwqT.a - uvwq.a) + DqqyB*(uvwqB.a - uvwq.a));`,
      ]
        .slice(0, numSpecies)
        .join("\n") +
      `\nfloat dq = 0.5*dot(dSquared,` +
      [`LDquU`, `LDqvV`, `LDqwW`, `LDqqQ`].slice(0, numSpecies).join(" + ") +
      `) + QFUN;\n`;
  }
  // Add the final line of the shader.
  switch (numSpecies) {
    case 1:
      shader += `result = vec4(du,0.0,0.0,0.0);`;
      break;
    case 2:
      shader += `result = vec4(du,dv,0.0,0.0);`;
      break;
    case 3:
      shader += `result = vec4(du,dv,dw,0.0);`;
      break;
    case 4:
      shader += `result = vec4(du,dv,dw,dq);`;
      break;
  }
  return (
    shader +
    `
    }`
  );
}

/**
 * Returns the shader code for updating the algebraic species in a reaction-diffusion simulation.
 * @returns {string} The shader code for updating the algebraic species.
 */
export function RDShaderAlgebraicSpecies() {
  return `
    updated.SPECIES = RHS.SPECIES / timescales.SPECIES;
    `;
}

/**
 * Returns the shader code for applying Dirichlet boundary conditions in the x-direction.
 * @param {string} [LR] - Optional argument to specify whether to return the shader code for the left boundary ("L"), right boundary ("R"), or both boundaries (undefined).
 * @returns {string} The shader code for applying Dirichlet boundary conditions in the x-direction.
 */
export function RDShaderDirichletX(LR) {
  const L = `
    if (textureCoords.x - step_x < 0.0) {
        updated.SPECIES = dirichletRHSSPECIESL;
    }
    `;
  const R = `
    if (textureCoords.x + step_x > 1.0) {
        updated.SPECIES = dirichletRHSSPECIESR;
    }
    `;
  if (LR == undefined) return L + R;
  if (LR == "L") return L;
  if (LR == "R") return R;
  return "";
}

/**
 * Returns the shader code for applying Dirichlet boundary conditions in the y-direction.
 * @param {string} [LR] - Optional argument to specify whether to return the shader code for the top boundary ("T"), bottom boundary ("B"), or both boundaries (undefined).
 * @returns {string} The shader code for applying Dirichlet boundary conditions in the y-direction.
 */
export function RDShaderDirichletY(TB) {
  const T = `
    if (textureCoords.y + step_y > 1.0) {
        updated.SPECIES = dirichletRHSSPECIEST;
    }
    `;
  const B = `
    if (textureCoords.y - step_y < 0.0) {
        updated.SPECIES = dirichletRHSSPECIESB;
    }
    `;
  if (TB == undefined) return T + B;
  if (TB == "T") return T;
  if (TB == "B") return B;
  return "";
}

/**
 * Returns a shader fragment that updates the SPECIES based on an indicator function.
 * @returns {string} The shader function as a string.
 */
export function RDShaderDirichletIndicatorFun() {
  return `
    if (float(indicatorFun) <= 0.0) {
        updated.SPECIES = `;
}

/**
 * Returns the final part of shader code for a reaction-diffusion simulation.
 * @returns {string} The shader code.
 */
export function RDShaderBot() {
  return ` 
    gl_FragColor = updated;
}`;
}

/**
 * Returns the bottom part of an MRT shader (no gl_FragColor assignment since
 * gl_FragColor and fragData1 are already written by the update code).
 * @returns {string} The shader code.
 */
export function RDShaderBotMRT() {
  return `
}`;
}

/**
 * Returns the top part of shader code for enforcing Dirichlet boundary conditions.
 * @returns {string} The shader code.
 */
export function RDShaderEnforceDirichletTop() {
  return `precision highp float;
    varying vec2 textureCoords;
    uniform sampler2D textureSource;
    uniform float dx;
    uniform float dy;
    uniform float L;
    uniform float L_x;
    uniform float L_y;
    uniform float L_min;
    uniform float t;
    uniform sampler2D imageSourceOne;
    uniform sampler2D imageSourceTwo;

    AUXILIARY_GLSL_FUNS

    const float ALPHA = 0.147;
    const float INV_ALPHA = 1.0 / ALPHA;
    const float BETA = 2.0 / (pi * ALPHA);
    float erfinv(float pERF) {
      float yERF;
      if (pERF == -1.0) {
        yERF = log(1.0 - (-0.99)*(-0.99));
      } else {
        yERF = log(1.0 - pERF*pERF);
      }
      float zERF = BETA + 0.5 * yERF;
      return sqrt(sqrt(zERF*zERF - yERF * INV_ALPHA) - zERF) * sign(pERF);
    }
    
    void main()
    {
        ivec2 texSize = textureSize(textureSource,0);
        float step_x = 1.0 / float(texSize.x);
        float step_y = 1.0 / float(texSize.y);
        float x = textureCoords.x * L_x + MINX;
        float y = textureCoords.y * L_y + MINY;
        float interior = float(textureCoords.x > 0.75*step_x && textureCoords.x < 1.0 - 0.75*step_x && textureCoords.y > 0.5*step_y && textureCoords.y < 1.0 - 0.75*step_y);
        float exterior = 1.0 - interior;

        vec4 uvwq = texture2D(textureSource, textureCoords);
        gl_FragColor = uvwq;
    `;
}

/**
 * Generates shader code for clamping species values to the edge of a texture in a given direction.
 * @param {string} direction - The direction in which to clamp the species values. Can include "H" for horizontal and/or "V" for vertical.
 * @returns {string} The generated GLSL code.
 */
export function clampSpeciesToEdgeShader(direction) {
  let out = "";
  if (direction.includes("H")) {
    out += `
    if (textureCoords.x - step_x < 0.0) {
      uvwqL.SPECIES = uvwq.SPECIES;
    }
    if (textureCoords.x + step_x > 1.0) {
      uvwqR.SPECIES = uvwq.SPECIES;
    }`;
  }
  if (direction.includes("V")) {
    out += `
    if (textureCoords.y + step_y > 1.0) {
      uvwqT.SPECIES = uvwq.SPECIES;
    }
    if (textureCoords.y - step_y < 0.0) {
      uvwqB.SPECIES = uvwq.SPECIES;
    }`;
  }
  return out;
}
