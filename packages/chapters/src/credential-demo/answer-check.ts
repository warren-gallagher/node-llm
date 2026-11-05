import { readFileSync } from "node:fs";
import type { FeatureExtractionPipeline } from "@huggingface/transformers";
import type { GptModel } from "@node-llm/core";
import type { Passage } from "../ch07-instruction/rag-corpus";
import { respond } from "../ch07-instruction/shared";
import {
  type AnswerRecord,
  generationChecks,
  indexChecks,
  provenanceCheck,
  rank,
  retrievalRerunCheck,
  textCheck,
} from "./answers";
import { type ClaimedFile, type DemoCheck, decodeJwtPayload } from "./checks";
import type { AqClient, VerificationResult } from "./client";
import { REPO_ROOT } from "./env";
import { digestMultibase, hashFile, niUri } from "./hashing";
import { embedPassages, type IndexEntry, passageSha256, RAG_DIR } from "./rag";
import type { AnswerRecordEntry, Manifest } from "./state";

/**
 * The buyer's check of one answer. The buyer holds the answer and question, a copy of the index,
 * the passages and the models, and checks them against the answer's DTE and the credentials it
 * links.
 */

/** What the buyer learns from the platform and the credentials, before looking at any file. */
export interface AnswerReading {
  answerUrl: string;
  platform: VerificationResult;
  record: AnswerRecord;
  /** The passages the collection passport lists, and who issued it. */
  listed: ClaimedFile[];
  collectionIssuer: string;
}

/** What the buyer holds. */
export interface AnswerFiles {
  question: string;
  answer: string;
  index: URL;
  /** Every passage text the buyer has, to rebuild the index and regenerate the answer. */
  passages: Passage[];
}

/** The answer with this ID, or the latest. */
export function pickAnswer(manifest: Manifest, answerId?: string): AnswerRecordEntry {
  const answers = manifest.rag?.answers ?? [];
  const found = answerId ? answers.find((a) => a.answerId === answerId) : answers.at(-1);
  if (!found) {
    throw new Error(
      answerId ? `no answer ${answerId} in the manifest` : "no answers yet: run pnpm demo:ask",
    );
  }
  return found;
}

export function isAnswerId(manifest: Manifest, id: string | undefined): boolean {
  return id !== undefined && (manifest.rag?.answers ?? []).some((a) => a.answerId === id);
}

/**
 * Asks the platform to verify the answer's DTE through the whole chain, reads what the DTE
 * records, and reads the collection passport (found in the platform's tree by the hash of the
 * collection the buyer holds).
 */
export async function readAnswer(client: AqClient, answerUrl: string): Promise<AnswerReading> {
  const platform = await client.verify("dte", answerUrl);
  const event = decodeJwtPayload(await client.fetchCredential(answerUrl)).credentialSubject[0];
  const inputs: { product: { id: string; name: string } }[] = event.inputProduct;
  // The answer DTE lists the question first, then the passages, then the index and the model.
  const record: AnswerRecord = {
    questionId: inputs[0]?.product.id ?? "",
    passageIds: inputs
      .filter((i) => i.product.name.startsWith("Passage "))
      .map((i) => i.product.id),
    indexId: inputs.at(-2)?.product.id ?? "",
    generatorId: inputs.at(-1)?.product.id ?? "",
    answerId: event.outputProduct[0]?.product.id ?? "",
  };

  const collectionId = niUri(
    hashFile(new URL("collection.json", RAG_DIR), "application/json").sha256,
  );
  const collection = findNode(
    platform,
    (node) => node.credentialType === "dpp" && node.subject === collectionId,
  );
  if (!collection) throw new Error(`no passport for the collection ${collectionId} in the chain`);
  const passport = decodeJwtPayload(await client.fetchCredential(collection.credentialId));
  return {
    answerUrl,
    platform,
    record,
    listed: passport.credentialSubject.characteristics.files,
    collectionIssuer: collection.issuer,
  };
}

function findNode(
  result: VerificationResult,
  matches: (node: VerificationResult) => boolean,
): VerificationResult | undefined {
  if (matches(result)) return result;
  for (const child of result.upstreamResults ?? []) {
    const found = findNode(child, matches);
    if (found) return found;
  }
  return undefined;
}

/** The checks that need only the credentials. */
export function answerReadingChecks(
  reading: AnswerReading,
  publisherDid: string | undefined,
): DemoCheck[] {
  return [
    {
      group: "retrieval",
      name: "The collection passport is Book Publisher's",
      passed: reading.collectionIssuer === publisherDid,
      detail: reading.collectionIssuer,
    },
    provenanceCheck(reading.record, reading.listed),
  ];
}

/** Every check: the credentials, then the buyer's files, rebuilding and rerunning what it can. */
export async function answerChecks(
  reading: AnswerReading,
  files: AnswerFiles,
  models: { embedder: FeatureExtractionPipeline; generator: GptModel; generatorSha256: string },
  publisherDid: string | undefined,
): Promise<DemoCheck[]> {
  const { record } = reading;
  const entries: IndexEntry[] = JSON.parse(readFileSync(files.index, "utf8")).entries;
  const indexSha256 = hashFile(files.index, "application/json").sha256;

  // Rebuild from the listed passages the buyer holds, each embedded on its own.
  const listed = new Set(reading.listed.map((file) => file.digestMultibase));
  const rebuildFrom = files.passages.filter((p) => listed.has(digestMultibase(passageSha256(p))));
  const rebuilt = await embedPassages(models.embedder, rebuildFrom);

  // Retrieve again from the buyer's copy of the index.
  const [question] = await embedPassages(models.embedder, [
    { id: "question", source: "", text: files.question },
  ]);
  const [best] = rank(entries, question?.vector ?? []);
  const retrieved = best
    ? [{ id: best.entry.passageId, sha256: best.entry.sha256, score: best.score }]
    : [];

  // Generate again from the recorded passage.
  const passageId = record.passageIds[0] ?? "";
  const passage = files.passages.find((p) => niUri(passageSha256(p)) === passageId);
  const regenerated = passage
    ? respond(
        models.generator,
        { instruction: files.question, input: passage.text, output: "" },
        256,
        true,
      )
    : "(the recorded passage isn't one the buyer holds)";

  return [
    ...answerReadingChecks(reading, publisherDid),
    textCheck("question", files.question, record.questionId),
    textCheck("answer", files.answer, record.answerId),
    ...indexChecks(record, indexSha256, entries, reading.listed, rebuilt),
    retrievalRerunCheck(record, retrieved),
    ...generationChecks(record, models.generatorSha256, regenerated),
  ];
}

/** The answer's texts, the deployer's index, and the passages, as the buyer holds them. */
export function answerFiles(
  entry: AnswerRecordEntry,
  manifest: Manifest,
  passages: Passage[],
): AnswerFiles {
  const index = manifest.rag?.index?.file;
  if (!index) throw new Error("no index in the manifest: run pnpm demo:rag-setup");
  return {
    question: entry.question,
    answer: entry.answer,
    index: new URL(index, REPO_ROOT),
    passages,
  };
}
