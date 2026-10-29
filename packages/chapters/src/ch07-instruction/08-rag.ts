import { env, type FeatureExtractionPipeline, pipeline } from "@huggingface/transformers";
import type { InstructionEntry } from "@node-llm/core";
import { CORPUS, type Passage, PLANTED, TRUSTED_SOURCES } from "./rag-corpus";
import { chosenSize, loadFineTuned, loadInstructionSplits, respond } from "./shared";

/**
 * Chapter 7, extra: retrieval-augmented generation (RAG).
 *
 * The fine-tuned model gets many facts wrong because GPT-2 learned few of them in
 * pre-training. RAG supplies the facts at answer time instead of training them in:
 *   1. an embedding model turns each passage of a small document collection into a vector;
 *   2. the question is turned into a vector the same way;
 *   3. the passage whose vector is closest (highest cosine similarity) is put into the
 *      prompt's `### Input:` section, and the model answers from it.
 * The lesson then plants a false passage in the collection to show why where documents
 * come from matters.
 *
 * The embedding model is all-MiniLM-L6-v2 (22.7 million parameters, 384-dimensional vectors),
 * run with Hugging Face's transformers.js and downloaded once (about 23 MB) into
 * `checkpoints/embeddings/`. Needs lesson 5's fine-tuned model.
 */
const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
/** The model repository's commit, so the vectors don't change if the repository does. */
const EMBEDDING_REVISION = "751bff37182d3f1213fa05d7196b954e230abad9";

/** Test-set questions (lesson 6's numbering, from 0) whose answers are in the collection. */
const QUESTIONS = [2, 13, 3, 1, 39, 24];

interface Indexed {
  passage: Passage;
  vector: number[];
}

async function main(): Promise<void> {
  const size = chosenSize();
  const model = loadFineTuned(size);
  if (!model) {
    const flag = size === "355m" ? " --355m" : "";
    console.log(`no fine-tuned GPT-2 ${size} yet: run lesson 5 first (pnpm ch07:5${flag})`);
    return;
  }
  const { test } = await loadInstructionSplits();
  const embedder = await loadEmbedder();

  const index = await buildIndex(embedder, CORPUS);
  console.log(`indexed ${index.length} passages as ${index[0]?.vector.length}-dimensional vectors`);

  for (const i of QUESTIONS) {
    const entry = test[i];
    if (!entry) continue;
    const hits = await search(embedder, index, entry.instruction);
    console.log(`\n${"=".repeat(70)}\n${entry.instruction}`);
    for (const hit of hits) console.log(`  ${hit.score.toFixed(3)}  ${hit.passage.text}`);
    console.log(`without retrieval: ${respond(model, entry, 256, true)}`);
    console.log(`with retrieval:    ${answerWith(model, entry, hits[0]?.passage)}`);
  }

  // A planted passage, worded to match the question closely.
  const entry = test[2] as InstructionEntry;
  const poisoned = [...index, ...(await buildIndex(embedder, [PLANTED]))];
  const [top] = await search(embedder, poisoned, entry.instruction);
  console.log(`\n${"=".repeat(70)}\nafter a false passage is planted in the collection:`);
  console.log(`  retrieved: "${top?.passage.text}" (source: ${top?.passage.source})`);
  console.log(`  answer:    ${answerWith(model, entry, top?.passage)}`);

  // Admission control: index only passages from trusted sources.
  const admitted = poisoned.filter((d) => TRUSTED_SOURCES.has(d.passage.source));
  const [trusted] = await search(embedder, admitted, entry.instruction);
  console.log("\nindexing only passages from trusted sources:");
  console.log(`  retrieved: "${trusted?.passage.text}"`);
  console.log(`  answer:    ${answerWith(model, entry, trusted?.passage)}`);

  model.dispose();
  await embedder.dispose();
}

/** The embedding model, downloaded once at a fixed revision. */
async function loadEmbedder(): Promise<FeatureExtractionPipeline> {
  env.cacheDir = new URL("../../../../checkpoints/embeddings/", import.meta.url).pathname;
  return pipeline("feature-extraction", EMBEDDING_MODEL, {
    revision: EMBEDDING_REVISION,
    dtype: "q8",
  });
}

/**
 * One vector per text: the model's token vectors averaged ("mean pooling"), then scaled to
 * length 1, so the dot product of two vectors is their cosine similarity.
 */
async function embed(embedder: FeatureExtractionPipeline, texts: string[]): Promise<number[][]> {
  const output = await embedder(texts, { pooling: "mean", normalize: true });
  return output.tolist() as number[][];
}

async function buildIndex(
  embedder: FeatureExtractionPipeline,
  passages: Passage[],
): Promise<Indexed[]> {
  const vectors = await embed(
    embedder,
    passages.map((p) => p.text),
  );
  return passages.map((passage, i) => ({ passage, vector: vectors[i] as number[] }));
}

/**
 * The `k` passages most similar to the query. A real system uses an approximate
 * nearest-neighbour index; with a dozen passages, comparing against every one is fine.
 */
async function search(
  embedder: FeatureExtractionPipeline,
  index: Indexed[],
  query: string,
  k = 3,
): Promise<{ passage: Passage; score: number }[]> {
  const [q] = await embed(embedder, [query]);
  return index
    .map((d) => ({ passage: d.passage, score: dot(q as number[], d.vector) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, x, i) => sum + x * (b[i] as number), 0);
}

/**
 * Answers with the passage as the prompt's input, the slot the template already has. The
 * prompt ends with the response heading (see `respond`), as it does for the answers without
 * retrieval, so the two are compared fairly.
 */
function answerWith(
  model: NonNullable<ReturnType<typeof loadFineTuned>>,
  entry: InstructionEntry,
  passage: Passage | undefined,
): string {
  return respond(model, { ...entry, input: passage?.text ?? "" }, 256, true);
}

main();
