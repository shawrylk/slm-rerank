// resource: the user cache folder of slm-rerank. It is never inside a searched repository, whose gates would see it.
import os from "node:os";
import path from "node:path";

export const CACHE_DIR_ENV = "SLM_RERANK_CACHE_DIR";

export function resolveCacheDir({ env = process.env, platform = process.platform, home = os.homedir() } = {}) {
  if (env[CACHE_DIR_ENV]) return env[CACHE_DIR_ENV];
  if (platform === "win32") return path.join(env.LOCALAPPDATA || path.join(home, "AppData", "Local"), "slm-rerank");
  if (platform === "darwin") return path.join(home, "Library", "Caches", "slm-rerank");
  return path.join(env.XDG_CACHE_HOME || path.join(home, ".cache"), "slm-rerank");
}
