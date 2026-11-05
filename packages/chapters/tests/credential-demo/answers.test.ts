import { describe, expect, it } from "vitest";
import { PLANTED } from "../../src/ch07-instruction/rag-corpus";
import {
  type AnswerRecord,
  generationChecks,
  indexChecks,
  provenanceCheck,
  rank,
  retrievalRerunCheck,
  textCheck,
  textDigest,
} from "../../src/credential-demo/answers";
import { answerDte } from "../../src/credential-demo/claims";
import { niUri, sha256Hex } from "../../src/credential-demo/hashing";
import {
  collectionPassages,
  type IndexEntry,
  passageFile,
  passageSha256,
} from "../../src/credential-demo/rag";
import { CTX, issued, MODEL } from "./fixtures";

// The builders return plain objects; tests read them loosely.
// biome-ignore lint/suspicious/noExplicitAny: test access into built claims
type Loose = any;

const passages = collectionPassages();
const listed = passages.map(passageFile);
const [austen, frost] = passages;
const QUESTION = "Name the author of 'Pride and Prejudice'.";
const ANSWER = "The novel was written by Jane Austen.";
const INDEX_SHA = "d".repeat(64);

function entry(passage: { id: string; text: string }, vector: number[]): IndexEntry {
  return { passageId: passage.id, sha256: sha256Hex(passage.text), vector };
}

const entries = [entry(austen, [1, 0]), entry(frost, [0, 1])];
const record: AnswerRecord = {
  questionId: niUri(sha256Hex(QUESTION)),
  answerId: niUri(sha256Hex(ANSWER)),
  passageIds: [niUri(passageSha256(austen))],
  generatorId: niUri(MODEL.weights.sha256),
  indexId: niUri(INDEX_SHA),
};

function failed(checks: { name: string; passed: boolean }[]): string[] {
  return checks.filter((check) => !check.passed).map((check) => check.name);
}

describe("an answer's record", () => {
  it("records the question, passage, index and model as inputs, and the answer as output", () => {
    const [event]: Loose = answerDte(
      CTX,
      {
        answerId: "a1",
        eventDate: "2026-11-05T12:00:00.000Z",
        question: { ...textDigest("question", QUESTION), text: QUESTION },
        answer: { ...textDigest("answer", ANSWER), text: ANSWER },
        passages: [{ ...textDigest(austen.id, austen.text), score: 0.759 }],
      },
      { name: MODEL.name, weights: MODEL.weights },
      { name: "Index", file: { ...MODEL.config, name: "index.json", sha256: INDEX_SHA } },
      { generatorPassport: issued("dpp", "generator"), indexPassport: issued("dpp", "index") },
    ).credentialSubject;
    expect(event.inputProduct.map((input: Loose) => input.product.id)).toEqual([
      record.questionId,
      record.passageIds[0],
      record.indexId,
      record.generatorId,
    ]);
    expect(event.inputProduct[1].product.name).toBe(`Passage ${austen.id}`);
    expect(event.outputProduct[0].product.id).toBe(record.answerId);
    expect(event.description).toContain("cosine similarity 0.759");
  });
});

describe("checking an answer", () => {
  it("matches the texts the buyer holds to the record", () => {
    expect(textCheck("answer", ANSWER, record.answerId).passed).toBe(true);
    expect(textCheck("answer", `${ANSWER} (edited)`, record.answerId).passed).toBe(false);
  });

  it("finds every passage the answer rests on in the collection passport", () => {
    expect(provenanceCheck(record, listed).passed).toBe(true);
    const planted = { ...record, passageIds: [niUri(passageSha256(PLANTED))] };
    expect(provenanceCheck(planted, listed)).toMatchObject({ passed: false });
  });

  it("accepts the recorded index, and catches an unlisted passage in it", () => {
    expect(failed(indexChecks(record, INDEX_SHA, entries, listed, entries))).toEqual([]);
    const planted = [...entries, entry(PLANTED, [0.7, 0.7])];
    expect(failed(indexChecks(record, "e".repeat(64), planted, listed, entries))).toEqual([
      "The index is the one the record names",
      "The index holds only passages the collection passport lists",
      "Rebuilding the index from its passages gives the same vectors",
    ]);
  });

  it("ranks passages by similarity, and checks the rerun returns the recorded one", () => {
    const [best] = rank(entries, [0.9, 0.1]);
    expect(best?.entry.passageId).toBe(austen.id);
    const rerun = (id: string, text: string) =>
      retrievalRerunCheck(record, [{ id, sha256: sha256Hex(text), score: 0.9 }]);
    expect(rerun(austen.id, austen.text).passed).toBe(true);
    expect(rerun(PLANTED.id, PLANTED.text).passed).toBe(false);
  });

  it("checks the regenerated answer against the record's hash, not the buyer's copy", () => {
    expect(failed(generationChecks(record, MODEL.weights.sha256, ANSWER))).toEqual([]);
    expect(failed(generationChecks(record, MODEL.weights.sha256, "Robert Frost."))).toEqual([
      "Generating again gives the recorded answer",
    ]);
    expect(failed(generationChecks(record, "0".repeat(64), ANSWER))).toEqual([
      "The answering model is the one the record names",
    ]);
  });
});
