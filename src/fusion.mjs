// The model can rank a chunk the query names below one it only resembles; the lexical z-score shifts its logit.

// A lexical outlier alone never lifts a no-signal chunk (0.05) past the default threshold (0.65).
export const LEXICAL_PRIOR_MAX = 3;
// One path hit, in lexical-score units: a smaller spread across the candidates is noise.
export const LEXICAL_SPREAD_FLOOR = 3;
const P_EPSILON = 1e-4;

const logit = p => {
  const x = Math.min(Math.max(p, P_EPSILON), 1 - P_EPSILON);
  return Math.log(x / (1 - x));
};
const sigmoid = x => 1 / (1 + Math.exp(-x));
const round4 = x => Math.round(x * 10000) / 10000;

/**
 * results[i] is the model's result for the chunk whose lexical score is lexicalScores[i].
 * population is the set a chunk is judged against: a caller that scores only the head of the
 * lexical order passes the lexical scores of every candidate, or the head is judged against itself.
 * Returns new results: `score` is fused, `rawScore` stays the model's own score.
 */
export function fuseLexicalPrior(results, lexicalScores, population = lexicalScores) {
  const n = lexicalScores.length;
  if (!n || n !== results.length) return results;
  const pool = population.length ? population : lexicalScores;

  const mean = pool.reduce((sum, x) => sum + x, 0) / pool.length;
  const sd = Math.sqrt(pool.reduce((sum, x) => sum + (x - mean) ** 2, 0) / pool.length);
  const spread = Math.max(sd, LEXICAL_SPREAD_FLOOR);

  return results.map((result, i) => {
    const lexicalScore = round4(lexicalScores[i]);
    // No model verdict means no evidence to shift; an unscored chunk must not surface as a hit.
    if (result.error) return { ...result, lexicalScore };
    const z = Math.max(-LEXICAL_PRIOR_MAX, Math.min(LEXICAL_PRIOR_MAX, (lexicalScores[i] - mean) / spread));
    return { ...result, score: round4(sigmoid(logit(result.rawScore) + z)), lexicalScore };
  });
}
