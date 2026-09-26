import { createRequire } from "node:module";
export { DEFAULT_RERANK_TOP, Reranker } from "./client.mjs";
export { applyTwoTierFilter, computeLexicalScore, getGitDiffFiles, rankLexical } from "./filter.mjs";
export { chunkFile, placeSlice, prepareCandidates, sliceContent } from "./chunker.mjs";
export { prepareCandidatesCached } from "./chunk-cache.mjs";
export { resolveCacheDir } from "./cache-dir.mjs";
export { appendUsage, readUsage, usageLogPath } from "./usage/log.mjs";
export { formatStats, summarizeUsage } from "./usage/stats.mjs";
export { isCodeQuestion, isMachineText } from "./hooks/code-question.mjs";
export { fuseLexicalPrior } from "./fusion.mjs";
export { stitchChunkContext } from "./stitcher.mjs";
export { attachGhostStubs, generateGhostStub } from "./stubber.mjs";
export { detectSlice, groupBySlice } from "./boundary.mjs";
export {
  autoDiscoverEndpoint,
  discoverCandidateFiles,
  extractQueryTerms,
  isTestPath,
  probePort,
  resolveEndpointEnv,
  resolveHostEnv,
  SLM_PORT_RANGE
} from "./discovery.mjs";
export { expandQuery, formatExpansionPrompt, parseExpansionText } from "./expander.mjs";
export { startMcpServer, handleMcpMessage } from "./mcp.mjs";

export const VERSION = createRequire(import.meta.url)("../package.json").version;
