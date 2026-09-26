import fs from "node:fs";

/** Adds `ghostStub` and `foldedLines` to each result, in place: `--stub` for the rerank and for `--fast`. */
export function attachGhostStubs(results) {
  for (const item of results) {
    const ghost = generateGhostStub(item.chunk.filePath, item.chunk);
    item.ghostStub = ghost.stub;
    item.foldedLines = ghost.foldedLines;
  }
  return results;
}

/**
 * Generate a concise "Ghost Stub" of the enclosing source file.
 * Preserves imports, types, and the full implementation of the target chunk,
 * while collapsing sibling function/method bodies into 1-line stubs.
 *
 * @param {string} filePath 
 * @param {{startLine: number, endLine: number, symbol?: string}} chunk 
 * @param {object} options
 * @returns {{stub: string, foldedLines: number, originalLines: number}}
 */
export function generateGhostStub(filePath, chunk, options = {}) {
  let content = options.content;
  if (!content) {
    try {
      content = fs.readFileSync(filePath, "utf-8");
    } catch {
      return { stub: chunk.content || "", foldedLines: 0, originalLines: 0 };
    }
  }

  const lines = content.split("\n");
  const totalLines = lines.length;
  const { startLine, endLine } = chunk;

  const resultLines = [];
  let foldedLines = 0;
  let inFoldableBlock = false;
  let blockStartLine = -1;
  let braceDepth = 0;
  let blockHeader = "";

  for (let i = 0; i < totalLines; i++) {
    const lineNum = i + 1;
    const line = lines[i];

    // Inside target chunk: always include verbatim with line numbers
    if (lineNum >= startLine && lineNum <= endLine) {
      if (inFoldableBlock) {
        // End any pending folded block before target chunk
        inFoldableBlock = false;
      }
      resultLines.push(`${lineNum}: ${line}`);
      continue;
    }

    // Keep top-of-file imports and requires verbatim
    if (lineNum < startLine && /^\s*(import\s|const\s+.*=\s*require\(|from\s+["']|export\s+(type|interface)\s)/.test(line)) {
      resultLines.push(`${lineNum}: ${line}`);
      continue;
    }

    // Detect function, method, or class block starts outside target chunk
    const isFuncStart = /^\s*(export\s+)?(async\s+)?(function\*?|class)\s+([a-zA-Z0-9_$]+)/.test(line) ||
                        /^\s*(public|private|protected|static|async)*\s*([a-zA-Z0-9_$]+)\s*\([^)]*\)\s*[:{]/.test(line);

    if (isFuncStart && !inFoldableBlock && line.includes("{")) {
      const openBraces = (line.match(/{/g) || []).length;
      const closeBraces = (line.match(/}/g) || []).length;

      if (openBraces > closeBraces) {
        inFoldableBlock = true;
        blockStartLine = lineNum;
        braceDepth = openBraces - closeBraces;
        // Keep the signature line
        blockHeader = line.replace(/\{.*$/, "").trimEnd();
        continue;
      }
    }

    if (inFoldableBlock) {
      const openBraces = (line.match(/{/g) || []).length;
      const closeBraces = (line.match(/}/g) || []).length;
      braceDepth += openBraces - closeBraces;

      if (braceDepth <= 0) {
        const count = lineNum - blockStartLine;
        foldedLines += count;
        resultLines.push(`${blockStartLine}: ${blockHeader} { /* ... [folded ${count} lines] ... */ }`);
        inFoldableBlock = false;
      }
      continue;
    }

    // Outside blocks and not in target chunk: keep types/interfaces, collapse whitespace runs
    if (/^\s*(export\s+)?(type|interface|enum)\s+/.test(line)) {
      resultLines.push(`${lineNum}: ${line}`);
    } else if (line.trim().length === 0) {
      if (resultLines.length > 0 && !resultLines[resultLines.length - 1].endsWith(": ")) {
        resultLines.push(`${lineNum}: `);
      }
    }
  }

  return {
    stub: resultLines.join("\n"),
    foldedLines,
    originalLines: totalLines
  };
}
