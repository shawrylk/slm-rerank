// Call-graph and type context stitching under <= 150 token cap

import fs from "node:fs";

export const MAX_STITCH_TOKENS = 150;

const ENCLOSING_TYPE_RE = /^(?:export\s+|public\s+)*(?:class|struct|interface|trait|enum)\s+([A-Za-z0-9_]+)/;

export function stitchChunkContext(chunk, maxTokens = MAX_STITCH_TOKENS) {
  if (!chunk.filePath || !fs.existsSync(chunk.filePath)) {
    return { text: "", tokenEst: 0 };
  }

  try {
    const content = fs.readFileSync(chunk.filePath, "utf-8");
    const lines = content.split("\n");
    const beforeLines = lines.slice(0, Math.max(0, chunk.startLine - 1));

    let enclosingType = null;
    for (let i = beforeLines.length - 1; i >= 0; i--) {
      const line = beforeLines[i].trim();
      const m = line.match(ENCLOSING_TYPE_RE);
      if (m) {
        enclosingType = line;
        break;
      }
    }

    if (!enclosingType) {
      return { text: "", tokenEst: 0 };
    }

    const text = `// [Stitched Context: Enclosing Type]\n// ${enclosingType}\n`;
    const tokenEst = Math.ceil(text.length / 4);
    if (tokenEst > maxTokens) {
      return { text: "", tokenEst: 0 };
    }

    return { text, tokenEst };
  } catch {
    return { text: "", tokenEst: 0 };
  }
}
