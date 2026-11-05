import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import { PLANTED } from "../../src/ch07-instruction/rag-corpus";
import {
  type CollectionFacts,
  collectionDpp,
  type ImportedModel,
  importedModelDpp,
  indexDpp,
  indexingDte,
  instructionModelDpp,
} from "../../src/credential-demo/claims";
import { niUri, sha256Hex } from "../../src/credential-demo/hashing";
import {
  admitPassages,
  collectionPassages,
  compareIndexes,
  INDEXING,
  passageFile,
  writeCollection,
} from "../../src/credential-demo/rag";
import { CTX, DATASET, DERIVATION, issued, MODEL } from "./fixtures";

// The builders return plain objects; tests read them loosely.
// biome-ignore lint/suspicious/noExplicitAny: test access into built claims
type Loose = any;

const LINK_TYPES = "https://test.uncefact.org/vocabulary/linkTypes/";
const passages = collectionPassages();

describe("the document collection", () => {
  it("lists each passage by the digest of its text", () => {
    const [first] = passages;
    expect(passageFile(first)).toEqual({
      name: first.id,
      mediaType: "text/plain",
      byteLength: Buffer.byteLength(first.text),
      digestMultibase: expect.stringMatching(/^uEi/),
    });
  });

  it("writes the same file, and so the same hash, every time", () => {
    const dir = (name: string) => pathToFileURL(`${mkdtempSync(join(tmpdir(), name))}/`);
    const a = writeCollection(passages, dir("collection-a-"));
    const b = writeCollection(passages, dir("collection-b-"));
    expect(a.sha256).toBe(b.sha256);
  });

  it("admits only the passages the collection passport lists", () => {
    const { admitted, rejected } = admitPassages([...passages, PLANTED], passages.map(passageFile));
    expect(admitted).toEqual(passages);
    expect(rejected).toEqual([PLANTED]);
  });

  it("rejects a listed passage whose text was changed", () => {
    const [first, ...rest] = passages;
    const changed = { ...first, text: `${first.text} (edited)` };
    const { rejected } = admitPassages([changed, ...rest], passages.map(passageFile));
    expect(rejected).toEqual([changed]);
  });
});

describe("comparing indexes", () => {
  const entry = (id: string, vector: number[], sha256 = sha256Hex(id)) => ({
    passageId: id,
    sha256,
    vector,
  });

  it("finds an identical rebuild", () => {
    const index = [entry("a", [0.6, 0.8]), entry("b", [1, 0])];
    expect(compareIndexes(index, structuredClone(index))).toEqual({
      missing: [],
      extra: [],
      changed: [],
      maxDifference: 0,
    });
  });

  it("finds a planted entry, a missing one, a changed text and drifting vectors", () => {
    const original = [entry("a", [0.6, 0.8]), entry("b", [1, 0]), entry("c", [0, 1])];
    const rebuilt = [
      entry("a", [0.6, 0.8001]),
      entry("b", [1, 0], "other"),
      entry("planted", [0.5, 0.5]),
    ];
    const comparison = compareIndexes(original, rebuilt);
    expect(comparison).toMatchObject({ missing: ["c"], extra: ["planted"], changed: ["b"] });
    expect(comparison.maxDifference).toBeCloseTo(1e-4, 8);
  });
});

describe("credential demo claims for retrieval", () => {
  const weights = {
    name: "model.safetensors",
    mediaType: "application/octet-stream",
    byteLength: 10,
    sha256: "a".repeat(64),
  };
  const imported: ImportedModel = {
    name: "GPT-2 124M (OpenAI)",
    description: "The base model.",
    architecture: "GPT-2",
    parameterCount: 124_439_808,
    modelConfig: {},
    files: [weights],
    source: {
      repository: "https://huggingface.co/openai-community/gpt2",
      revision: "607a30d783dfa663caf39e06633721c8d4cfcd7e",
      publishedSha256: weights.sha256,
      licence: { id: "https://huggingface.co/openai-community/gpt2", name: "MIT" },
      producer: {
        name: "OpenAI, as published on Hugging Face",
        country: { countryCode: "US", countryName: "United States of America" },
      },
    },
  };
  const collection: CollectionFacts = {
    title: "Reference passages",
    description: "Passages.",
    file: {
      name: "collection.json",
      mediaType: "application/json",
      byteLength: 2000,
      sha256: "c".repeat(64),
    },
    passages: passages.map(passageFile),
    licence: { id: "https://github.com/example/node-llm", name: "No licence stated" },
  };
  const index = {
    name: "Reference passages index",
    file: {
      name: "index.json",
      mediaType: "application/json",
      byteLength: 5000,
      sha256: "d".repeat(64),
    },
    passageCount: passages.length,
  };
  const collectionPassport = issued("dpp", "collection");
  const embedderPassport = issued("dpp", "embedder");

  it("says an import passport attests the source, not how the model was made", () => {
    const passport: Loose = importedModelDpp(CTX, imported).credentialSubject;
    expect(passport.id).toBe(niUri(weights.sha256));
    expect(passport.description).toContain("matching the SHA-256 it publishes");
    expect(passport.description).toContain("not how the model was made");
    expect(passport.characteristics.source.revision).toBe(imported.source.revision);
  });

  it("links the answering model to its fine-tuning and its imported parent", () => {
    const fineTuning = issued("dte", "fine-tuning");
    const parentPassport = issued("dpp", "gpt2");
    const rights = issued("dcc", "instruction-rights");
    const passport: Loose = instructionModelDpp(CTX, MODEL, {
      fineTuning,
      parentName: imported.name,
      parentPassport,
      data: { dataset: DATASET, rights },
    }).credentialSubject;
    expect(passport.relatedDocument.map((l: Loose) => l.linkURL)).toEqual([
      fineTuning.credentialId,
      parentPassport.credentialId,
    ]);
    expect(passport.performanceClaim[0].evidence[0].linkURL).toBe(rights.credentialId);
  });

  it("lists every passage in the collection passport", () => {
    const passport: Loose = collectionDpp(CTX, collection).credentialSubject;
    expect(passport.id).toBe(niUri(collection.file.sha256));
    expect(passport.characteristics.files).toHaveLength(passages.length + 1);
  });

  it("records indexing from the collection and the embedder to the index", () => {
    const [event]: Loose = indexingDte(
      CTX,
      collection,
      imported,
      index,
      { ...DERIVATION, processType: "Embedding index", hyperparameters: INDEXING },
      {
        collectionPassport,
        embedderPassport,
      },
    ).credentialSubject;
    expect(event.inputProduct.map((p: Loose) => p.product.id)).toEqual([
      niUri(collection.file.sha256),
      niUri(weights.sha256),
    ]);
    expect(event.outputProduct[0].product.id).toBe(niUri(index.file.sha256));
    expect(event.relatedDocument.map((l: Loose) => l.linkType)).toEqual([
      `${LINK_TYPES}dpp`,
      `${LINK_TYPES}dpp`,
    ]);
    expect(event.trainingRun.hyperparameters.batching).toBe("each passage embedded on its own");
  });

  it("links the index passport to its indexing and its collection", () => {
    const indexing = issued("dte", "indexing");
    const passport: Loose = indexDpp(
      CTX,
      { name: index.name, description: "An index.", file: index.file },
      { indexing, collectionPassport, collectionTitle: collection.title },
    ).credentialSubject;
    expect(passport.id).toBe(niUri(index.file.sha256));
    expect(passport.relatedDocument.map((l: Loose) => l.linkURL)).toEqual([
      indexing.credentialId,
      collectionPassport.credentialId,
    ]);
  });
});
