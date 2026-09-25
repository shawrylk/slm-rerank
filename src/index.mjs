import { createRequire } from "node:module";
export { Reranker } from "./client.mjs";
export { applyTwoTierFilter, computeLexicalScore, getGitDiffFiles } from "./filter.mjs";
export { chunkFile, prepareCandidates } from "./chunker.mjs";
export { fuseLexicalPrior } from "./fusion.mjs";
export { stitchChunkContext } from "./stitcher.mjs";
export { generateGhostStub } from "./stubber.mjs";
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
