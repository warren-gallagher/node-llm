import { mkdirSync, writeFileSync } from "node:fs";
import { env, type FeatureExtractionPipeline, pipeline } from "@huggingface/transformers";
import { CORPUS, type Passage } from "../ch07-instruction/rag-corpus";
import type { ClaimedFile } from "./checks";
import { REPO_ROOT } from "./env";
import { digestMultibase, type FileDigest, hashFile, sha256Hex } from "./hashing";

/**
 * Retrieval for the credential demo's phase 3: chapter 7's passages as a credentialed document
 * collection, and an index of their embeddings that anyone holding the passages and the
 * embedding model can rebuild.
 *
 * Each passage is embedded on its own. Embedded in a batch, a passage's vector depends on the
 * other passages in the batch (by up to 0.018 here), so adding one passage would change every
 * other vector; on its own, a passage's vector depends only on its text.
 */

export const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
/** The model repository's commit chapter 7 pins. */
export const EMBEDDING_REVISION = "751bff37182d3f1213fa05d7196b954e230abad9";
export const EMBEDDINGS_DIR = new URL("checkpoints/embeddings/", REPO_ROOT);
export const RAG_DIR = new URL("checkpoints/verdict-demo/rag/", REPO_ROOT);

/** How the index is built, enough to rebuild it. */
export const INDEXING = {
  embeddingModel: EMBEDDING_MODEL,
  revision: EMBEDDING_REVISION,
  dtype: "q8",
  pooling: "mean",
  normalize: true,
  batching: "each passage embedded on its own",
} as const;

/** The passages chapter 7 wrote for its collection; its planted passage isn't one of them. */
export function collectionPassages(): Passage[] {
  return CORPUS;
}

export function passageSha256(passage: Passage): string {
  return sha256Hex(passage.text);
}

/** A passage as the collection passport lists it: by id, size and the digest of its text. */
export function passageFile(passage: Passage): ClaimedFile & { mediaType: string } {
  return {
    name: passage.id,
    mediaType: "text/plain",
    byteLength: Buffer.byteLength(passage.text, "utf8"),
    digestMultibase: digestMultibase(passageSha256(passage)),
  };
}

/** Writes the collection as one file, `[{ id, text }]` in order, and hashes it. */
export function writeCollection(passages: Passage[], dir: URL = RAG_DIR): FileDigest {
  mkdirSync(dir, { recursive: true });
  const path = new URL("collection.json", dir);
  writeFileSync(path, JSON.stringify(passages.map(({ id, text }) => ({ id, text }))));
  return hashFile(path, "application/json");
}

/**
 * Admission control: the candidate passages a verified collection passport lists, by the digest
 * of their text. Anything else, such as a planted passage, is rejected.
 */
export function admitPassages(
  candidates: Passage[],
  listed: ClaimedFile[],
): { admitted: Passage[]; rejected: Passage[] } {
  const digests = new Set(listed.map((file) => file.digestMultibase));
  const admitted = candidates.filter((p) => digests.has(digestMultibase(passageSha256(p))));
  const rejected = candidates.filter((p) => !admitted.includes(p));
  return { admitted, rejected };
}

export interface IndexEntry {
  passageId: string;
  /** SHA-256 (hex) of the passage's text. */
  sha256: string;
  vector: number[];
}

/** The embedding model as chapter 7 loads it, from the local cache, at the pinned revision. */
export async function loadEmbedder(): Promise<FeatureExtractionPipeline> {
  env.cacheDir = EMBEDDINGS_DIR.pathname;
  return pipeline("feature-extraction", EMBEDDING_MODEL, {
    revision: EMBEDDING_REVISION,
    dtype: INDEXING.dtype,
  });
}

/** One normalized, mean-pooled vector per passage, each passage embedded on its own. */
export async function embedPassages(
  embedder: FeatureExtractionPipeline,
  passages: Passage[],
): Promise<IndexEntry[]> {
  const entries: IndexEntry[] = [];
  for (const passage of passages) {
    const output = await embedder([passage.text], {
      pooling: INDEXING.pooling,
      normalize: INDEXING.normalize,
    });
    const [vector] = output.tolist() as number[][];
    entries.push({ passageId: passage.id, sha256: passageSha256(passage), vector });
  }
  return entries;
}

/** Writes the index file, `{ indexing, embeddingModelId, entries }`, and hashes it. */
export function writeIndex(
  entries: IndexEntry[],
  embeddingModelId: string,
  dir: URL = RAG_DIR,
): FileDigest {
  mkdirSync(dir, { recursive: true });
  const path = new URL("index.json", dir);
  writeFileSync(path, JSON.stringify({ indexing: INDEXING, embeddingModelId, entries }));
  return hashFile(path, "application/json");
}

/** How a rebuilt index differs from an original, entry by entry. */
export interface IndexComparison {
  /** Passages in the original that the rebuild lacks, and the other way round. */
  missing: string[];
  extra: string[];
  /** Passages whose text differs (by digest) between the two. */
  changed: string[];
  /** The largest difference in any one vector value, over passages in both. */
  maxDifference: number;
}

export function compareIndexes(original: IndexEntry[], rebuilt: IndexEntry[]): IndexComparison {
  const byId = new Map(rebuilt.map((entry) => [entry.passageId, entry]));
  const comparison: IndexComparison = { missing: [], extra: [], changed: [], maxDifference: 0 };
  for (const entry of original) {
    const other = byId.get(entry.passageId);
    if (!other) {
      comparison.missing.push(entry.passageId);
    } else if (other.sha256 !== entry.sha256) {
      comparison.changed.push(entry.passageId);
    } else {
      for (const [i, value] of entry.vector.entries()) {
        comparison.maxDifference = Math.max(
          comparison.maxDifference,
          Math.abs(value - (other.vector[i] ?? Number.NaN)),
        );
      }
    }
    byId.delete(entry.passageId);
  }
  comparison.extra = [...byId.keys()];
  return comparison;
}
