import { readFileSync } from "node:fs";
import type { FeatureExtractionPipeline } from "@huggingface/transformers";
import type { GptModel, InstructionEntry } from "@node-llm/core";
import { CORPUS, type Passage, PLANTED } from "../ch07-instruction/rag-corpus";
import { respond } from "../ch07-instruction/shared";
import type { ClaimedFile, DemoCheck } from "./checks";
import { digestMultibase, type FileDigest, niUri, sha256Hex } from "./hashing";
import { compareIndexes, embedPassages, type IndexEntry, passageSha256 } from "./rag";

/**
 * Answers for the credential demo's phase 3: Model Deployment Inc. answers a question with the
 * passage it retrieves, and records the answer in a DTE. A buyer holding the answer, the index,
 * the passages and the models checks it against that record: every passage it rests on is listed
 * in the publisher's collection passport, the index holds only such passages, and retrieving and
 * generating again give the same passage and answer.
 */

/** The largest vector difference a rebuilt index may show, across machines. */
export const INDEX_TOLERANCE = 1e-5;

/** One retrieved passage, as the answer record keeps it. */
export interface RetrievedPassage {
  id: string;
  sha256: string;
  score: number;
}

/** The facts of one answer: the question, the passage retrieved, and the text generated. */
export interface AnswerFacts {
  question: string;
  answer: string;
  passages: RetrievedPassage[];
}

/** A text as a product: identified by the SHA-256 of its UTF-8 bytes. */
export function textDigest(name: string, text: string): FileDigest {
  return {
    name,
    mediaType: "text/plain",
    byteLength: Buffer.byteLength(text, "utf8"),
    sha256: sha256Hex(text),
  };
}

/** The index entries in order of cosine similarity to the question (vectors are normalized). */
export function rank(
  entries: IndexEntry[],
  questionVector: number[],
): { entry: IndexEntry; score: number }[] {
  return entries
    .map((entry) => ({ entry, score: dot(questionVector, entry.vector) }))
    .sort((a, b) => b.score - a.score);
}

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] ?? 0) * (b[i] ?? 0);
  return sum;
}

/** The passage best matching a question, and the answer the model generates from it. */
export async function answer(
  embedder: FeatureExtractionPipeline,
  generator: GptModel,
  entries: IndexEntry[],
  passages: Passage[],
  entry: InstructionEntry,
): Promise<AnswerFacts> {
  const [question] = await embedPassages(embedder, [
    { id: "question", source: "", text: entry.instruction },
  ]);
  const [best] = rank(entries, question?.vector ?? []);
  const passage = passages.find((p) => p.id === best?.entry.passageId);
  if (!best || !passage) throw new Error("the index is empty");
  return {
    question: entry.instruction,
    answer: respond(generator, { ...entry, input: passage.text }, 256, true),
    passages: [{ id: passage.id, sha256: best.entry.sha256, score: best.score }],
  };
}

/** Every passage a buyer might hold: the collection's, and chapter 7's planted one. */
export function knownPassages(): Passage[] {
  return [...CORPUS, PLANTED];
}

/** The index file's entries. */
export function readIndex(path: URL): IndexEntry[] {
  return JSON.parse(readFileSync(path, "utf8")).entries;
}

// ============================================================================
// The buyer's checks of one answer
// ============================================================================

/** What an answer's DTE records, by id. */
export interface AnswerRecord {
  questionId: string;
  answerId: string;
  passageIds: string[];
  generatorId: string;
  indexId: string;
}

/** The text the buyer holds is the text the record names. */
export function textCheck(
  kind: "question" | "answer",
  text: string,
  recordedId: string,
): DemoCheck {
  const sha256 = sha256Hex(text);
  return {
    group: "answer",
    name: `The ${kind} is the one the record names`,
    passed: niUri(sha256) === recordedId,
    detail: `sha256 ${sha256}`,
  };
}

/** Every passage the answer rests on is listed, by digest, in the collection passport. */
export function provenanceCheck(record: AnswerRecord, listed: ClaimedFile[]): DemoCheck {
  const digests = new Set(listed.map((file) => niUri(digestToHex(file.digestMultibase))));
  const unlisted = record.passageIds.filter((id) => !digests.has(id));
  return {
    group: "retrieval",
    name: "Every passage the answer rests on is in the publisher's collection passport",
    passed: record.passageIds.length > 0 && unlisted.length === 0,
    detail:
      unlisted.length === 0
        ? `${record.passageIds.length} passage(s), all listed`
        : `not listed: ${unlisted.join(", ")}`,
  };
}

/**
 * The buyer's copy of the index is the one the record names, holds only listed passages, and
 * matches a rebuild from those passages.
 */
export function indexChecks(
  record: AnswerRecord,
  indexSha256: string,
  entries: IndexEntry[],
  listed: ClaimedFile[],
  rebuilt: IndexEntry[],
): DemoCheck[] {
  const digests = new Set(listed.map((file) => file.digestMultibase));
  const unlisted = entries.filter((entry) => !digests.has(digestMultibase(entry.sha256)));
  const comparison = compareIndexes(entries, rebuilt);
  const exact =
    comparison.missing.length + comparison.extra.length + comparison.changed.length === 0 &&
    comparison.maxDifference <= INDEX_TOLERANCE;
  return [
    {
      group: "retrieval",
      name: "The index is the one the record names",
      passed: niUri(indexSha256) === record.indexId,
      detail: `sha256 ${indexSha256}`,
    },
    {
      group: "retrieval",
      name: "The index holds only passages the collection passport lists",
      passed: unlisted.length === 0,
      detail:
        unlisted.length === 0
          ? `${entries.length} passages, all listed`
          : `not listed: ${unlisted.map((entry) => entry.passageId).join(", ")}`,
    },
    {
      group: "retrieval",
      name: "Rebuilding the index from its passages gives the same vectors",
      passed: exact,
      detail:
        `largest difference ${comparison.maxDifference} (tolerance ${INDEX_TOLERANCE})` +
        (comparison.missing.length
          ? `; in the index but not the rebuild: ${comparison.missing.join(", ")}`
          : "") +
        (comparison.extra.length
          ? `; in the rebuild but not the index: ${comparison.extra.join(", ")}`
          : "") +
        (comparison.changed.length ? `; text differs: ${comparison.changed.join(", ")}` : ""),
    },
  ];
}

/** Retrieving again returns the passages the record names. */
export function retrievalRerunCheck(
  record: AnswerRecord,
  retrieved: RetrievedPassage[],
): DemoCheck {
  const ids = retrieved.map((passage) => niUri(passage.sha256));
  const same =
    ids.length === record.passageIds.length && ids.every((id, i) => id === record.passageIds[i]);
  return {
    group: "retrieval",
    name: "Retrieving again returns the recorded passage",
    passed: same,
    detail: retrieved.map((p) => `${p.id} (similarity ${p.score.toFixed(3)})`).join(", "),
  };
}

/**
 * Generating again, with the recorded model and passage, gives the answer the record names, by
 * hash; so this doesn't depend on the buyer's copy of the answer.
 */
export function generationChecks(
  record: AnswerRecord,
  generatorSha256: string,
  regenerated: string,
): DemoCheck[] {
  const same = niUri(sha256Hex(regenerated)) === record.answerId;
  return [
    {
      group: "generation",
      name: "The answering model is the one the record names",
      passed: niUri(generatorSha256) === record.generatorId,
      detail: `sha256 ${generatorSha256}`,
    },
    {
      group: "generation",
      name: "Generating again gives the recorded answer",
      passed: same,
      detail: same ? `"${regenerated}"` : `regenerated a different answer: "${regenerated}"`,
    },
  ];
}

/** The hex SHA-256 inside a `digestMultibase` (base64url multihash). */
function digestToHex(value: string): string {
  return Buffer.from(value.slice(1), "base64url").subarray(2).toString("hex");
}

/** The passage whose text has this digest, among those the buyer holds. */
export function passageBySha(passages: Passage[], sha256: string): Passage | undefined {
  return passages.find((passage) => passageSha256(passage) === sha256);
}
