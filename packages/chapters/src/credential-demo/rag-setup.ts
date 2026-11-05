import { closeSync, openSync, readFileSync, readSync, statSync } from "node:fs";
import { GPT2_CONFIG, GPT2_MODELS } from "../ch05-training/gpt2";
import { PLANTED } from "../ch07-instruction/rag-corpus";
import type { ClaimedFile } from "./checks";
import {
  type CollectionFacts,
  collectionDpp,
  type DemoContext,
  type DerivationFacts,
  derivationDte,
  type ImportedModel,
  importedModelDpp,
  indexDpp,
  indexingDte,
  instructionModelDpp,
  type ModelFacts,
} from "./claims";
import { AqClient, type CredentialType, type SignRequest } from "./client";
import { loadDemoEnv, REPO_ROOT } from "./env";
import { type FileDigest, hashFile, niUri } from "./hashing";
import type { PartyKey } from "./parties";
import {
  authorityId,
  datasetFacts,
  gitCode,
  instructionFacts,
  printVerification,
} from "./pipeline";
import {
  admitPassages,
  collectionPassages,
  compareIndexes,
  EMBEDDING_MODEL,
  EMBEDDING_REVISION,
  EMBEDDINGS_DIR,
  embedPassages,
  INDEXING,
  loadEmbedder,
  passageFile,
  RAG_DIR,
  writeCollection,
  writeIndex,
} from "./rag";
import {
  type IssuedCredential,
  loadManifest,
  type Manifest,
  partyIdentities,
  type RagRecord,
  saveManifest,
  toIssued,
} from "./state";

/**
 * Credential demo, phase 3, step 1: the retrieval system's credentials.
 *
 * 1. Model Publisher's import passports for the two models from outside: OpenAI's GPT-2 124M and
 *    the all-MiniLM-L6-v2 embedding model at the revision chapter 7 pins.
 * 2. The instruction-tuned GPT-2 that answers questions (chapter 7, lesson 5): a fine-tuning DTE
 *    from GPT-2 on the instruction data, and a passport.
 * 3. Book Publisher's passport for chapter 7's passages, each listed by the digest of its text.
 * 4. Admission control: Model Deployment Inc. indexes only passages a verified collection passport
 *    from Book Publisher lists. Chapter 7's planted passage isn't listed, so it stays out.
 * 5. The index: each admitted passage embedded on its own, an indexing DTE and a passport from
 *    Model Deployment Inc., and a rebuild to show anyone holding the passages and the embedding
 *    model gets the same index.
 *
 * Each credential is reused while the file it describes is unchanged.
 *
 * Usage: pnpm demo:rag-setup [--dry-run]
 */

const GPT2_NAME = "GPT-2 124M (OpenAI)";
const EMBEDDER_NAME = "all-MiniLM-L6-v2 (Sentence Transformers, ONNX int8 export)";
const GENERATOR_NAME = "GPT-2 124M instruction-tuned (chapter 7)";
const GENERATOR_CHECKPOINT = new URL("checkpoints/gpt2-124m-instruct/", REPO_ROOT);
const COLLECTION_TITLE = "Reference passages (chapter 7 retrieval lesson)";
const INDEX_NAME = "Reference passages index (all-MiniLM-L6-v2)";

async function main(): Promise<void> {
  const env = loadDemoEnv();
  const client = new AqClient(env);
  const manifest = loadManifest(env.apiUrl, env.tenantId);
  const dryRun = process.argv.includes("--dry-run");
  const fineTuningData = manifest.fineTuningDataset;
  if (!fineTuningData) {
    throw new Error("the instruction data has no credentials yet: run pnpm demo:finetune");
  }
  const ctx: DemoContext = {
    apiUrl: env.apiUrl,
    parties: partyIdentities(manifest),
    date: new Date().toISOString().slice(0, 10),
  };
  const code = gitCode();
  const rag: RagRecord = { ...manifest.rag };
  if (dryRun) console.log("Dry run: the platform validates and signs, but publishes nothing\n");

  /** Signs a request, or returns the record kept for the same file. */
  async function issue(
    kept: { sha256: string; passport: IssuedCredential } | undefined,
    sha256: string,
    label: string,
    signer: PartyKey,
    type: CredentialType,
    request: () => SignRequest,
  ): Promise<IssuedCredential> {
    if (kept?.sha256 === sha256) {
      console.log(`  ${label}: reusing ${kept.passport.credentialId}`);
      return kept.passport;
    }
    const issued = toIssued(
      await client.sign(type, authorityId(manifest, signer), request(), { store: !dryRun }),
    );
    console.log(`  ${label}: ${issued.credentialId}`);
    return issued;
  }

  // 1. The models from outside.
  console.log("Models from outside (Model Publisher's import passports)");
  const gpt2 = await gpt2Import();
  const gpt2Weights = gpt2.files[0] as FileDigest;
  const gpt2Passport = await issue(
    rag.gpt2,
    gpt2Weights.sha256,
    GPT2_NAME,
    "modelPublisher",
    "dpp",
    () => importedModelDpp(ctx, gpt2),
  );
  rag.gpt2 = { sha256: gpt2Weights.sha256, passport: gpt2Passport };
  const embedder = await embedderImport();
  const embedderWeights = embedder.files[0] as FileDigest;
  const embedderPassport = await issue(
    rag.embedder,
    embedderWeights.sha256,
    EMBEDDER_NAME,
    "modelPublisher",
    "dpp",
    () => importedModelDpp(ctx, embedder),
  );
  rag.embedder = { sha256: embedderWeights.sha256, passport: embedderPassport };

  // 2. The instruction-tuned model that answers.
  console.log("\nThe answering model (Model Publisher)");
  const instructions = await instructionFacts(datasetFacts());
  if (instructions.file.sha256 !== fineTuningData.sha256) {
    throw new Error("the instruction data changed since its credentials were issued");
  }
  const generator = generatorFacts();
  if (rag.generator?.sha256 === generator.weights.sha256) {
    console.log(`  ${GENERATOR_NAME}: reusing ${rag.generator.passport.credentialId}`);
  } else {
    const parent = { name: GPT2_NAME, weights: gpt2Weights, parameterCount: gpt2.parameterCount };
    const fineTuning = toIssued(
      await client.sign(
        "dte",
        authorityId(manifest, "modelPublisher"),
        derivationDte(ctx, parent, generator, generatorRun(code), {
          parentPassport: gpt2Passport,
          data: {
            dataset: instructions,
            passport: fineTuningData.passport,
            rights: fineTuningData.rights,
          },
        }),
        { store: !dryRun },
      ),
    );
    console.log(`  fine-tuning DTE: ${fineTuning.credentialId}`);
    const passport = toIssued(
      await client.sign(
        "dpp",
        authorityId(manifest, "modelPublisher"),
        instructionModelDpp(ctx, generator, {
          fineTuning,
          parentName: GPT2_NAME,
          parentPassport: gpt2Passport,
          data: { dataset: instructions, rights: fineTuningData.rights },
        }),
        { store: !dryRun },
      ),
    );
    console.log(`  ${GENERATOR_NAME}: ${passport.credentialId}`);
    rag.generator = { sha256: generator.weights.sha256, passport, fineTuning };
  }

  // 3. The document collection.
  console.log("\nThe document collection (Book Publisher)");
  const passages = collectionPassages();
  const collection: CollectionFacts = {
    title: COLLECTION_TITLE,
    description:
      `${passages.length} short reference passages written for chapter 7's retrieval lesson, ` +
      "as JSON. Each passage is listed by the digest of its text, so a retrieval system can " +
      "admit exactly these passages.",
    file: writeCollection(passages),
    passages: passages.map(passageFile),
    licence: { id: code.repository, name: "No licence stated: written for this repository" },
  };
  const collectionPassport = await issue(
    rag.collection,
    collection.file.sha256,
    COLLECTION_TITLE,
    "bookPublisher",
    "dpp",
    () => collectionDpp(ctx, collection),
  );
  rag.collection = { sha256: collection.file.sha256, passport: collectionPassport };

  // 4. Admission control, as Model Deployment Inc.
  console.log("\nAdmission control (Model Deployment Inc.)");
  const listed = dryRun
    ? collection.passages
    : await verifiedListing(client, manifest, collectionPassport);
  const { admitted, rejected } = admitPassages([...passages, PLANTED], listed);
  console.log(`  admitted ${admitted.length} passages listed in the collection passport`);
  for (const passage of rejected) {
    console.log(`  rejected "${passage.text}" (${passage.id}): no collection passport lists it`);
  }

  // 5. The index.
  console.log("\nThe index (Model Deployment Inc.)");
  const eventDate = new Date().toISOString();
  const started = performance.now();
  const model = await loadEmbedder();
  const entries = await embedPassages(model, admitted);
  const durationSeconds = Math.round((performance.now() - started) / 100) / 10;
  const indexFile = writeIndex(entries, niUri(embedderWeights.sha256));
  console.log(
    `  ${entries.length} vectors of ${entries[0]?.vector.length} dimensions, sha256 ${indexFile.sha256}`,
  );
  const rebuilt = compareIndexes(entries, await embedPassages(model, admitted));
  await model.dispose();
  console.log(`  rebuilt from the same passages: largest difference ${rebuilt.maxDifference}`);

  if (rag.index?.sha256 === indexFile.sha256) {
    console.log(`  reusing ${rag.index.passport.credentialId}`);
  } else {
    const indexId = `${eventDate.slice(0, 19).replaceAll(":", "-")}Z-index`;
    const index = { name: INDEX_NAME, file: indexFile, passageCount: entries.length };
    const indexing = toIssued(
      await client.sign(
        "dte",
        authorityId(manifest, "modelDeployer"),
        indexingDte(
          ctx,
          collection,
          embedder,
          index,
          {
            derivationId: indexId,
            eventDate,
            processType: "Embedding index",
            hyperparameters: INDEXING,
            code,
            durationSeconds,
          },
          { collectionPassport, embedderPassport },
        ),
        { store: !dryRun },
      ),
    );
    console.log(`  indexing DTE: ${indexing.credentialId}`);
    const passport = toIssued(
      await client.sign(
        "dpp",
        authorityId(manifest, "modelDeployer"),
        indexDpp(
          ctx,
          {
            name: INDEX_NAME,
            description:
              `${entries.length} passages admitted from ${COLLECTION_TITLE}, each embedded on its ` +
              `own with ${EMBEDDER_NAME} (mean pooling, normalized).`,
            file: indexFile,
          },
          { indexing, collectionPassport, collectionTitle: COLLECTION_TITLE },
        ),
        { store: !dryRun },
      ),
    );
    console.log(`  index passport: ${passport.credentialId}`);
    rag.index = {
      sha256: indexFile.sha256,
      passport,
      indexing,
      file: new URL("index.json", RAG_DIR).pathname.replace(new URL(".", REPO_ROOT).pathname, ""),
    };
  }

  if (dryRun) {
    console.log("\nDry run complete: every credential passed the platform's validation.");
    return;
  }
  manifest.rag = rag;
  saveManifest(manifest);

  console.log("\nPlatform verification:");
  for (const credential of [rag.generator?.passport, rag.index?.passport]) {
    if (credential)
      printVerification(await client.verify(credential.type, credential.credentialId));
  }
}

/** OpenAI's GPT-2 124M as chapter 5 downloaded it from Hugging Face. */
async function gpt2Import(): Promise<ImportedModel> {
  const { repo } = GPT2_MODELS["124m"];
  const path = new URL("checkpoints/gpt2-124m/model.safetensors", REPO_ROOT);
  const info = await huggingFace(`api/models/${repo}`);
  const tree = await huggingFace(`api/models/${repo}/tree/main`);
  return {
    name: GPT2_NAME,
    description: "OpenAI's GPT-2 small, the base model chapters 5 to 7 load.",
    architecture: "GPT-2",
    parameterCount: safetensorsParameterCount(path),
    modelConfig: GPT2_CONFIG,
    files: [hashFile(path, "application/octet-stream")],
    source: {
      repository: `https://huggingface.co/${repo}`,
      revision: info.sha,
      publishedSha256: tree.find((file: { path: string }) => file.path === "model.safetensors")?.lfs
        ?.oid,
      licence: { id: `https://huggingface.co/${repo}`, name: "MIT (as the model card lists it)" },
      producer: {
        name: "OpenAI, as published on Hugging Face",
        country: { countryCode: "US", countryName: "United States of America" },
      },
    },
  };
}

/** The embedding model as chapter 7 loads it, from the local cache at the pinned revision. */
async function embedderImport(): Promise<ImportedModel> {
  const dir = new URL(`${EMBEDDING_MODEL}/${EMBEDDING_REVISION}/`, EMBEDDINGS_DIR);
  const tree = await huggingFace(`api/models/${EMBEDDING_MODEL}/tree/${EMBEDDING_REVISION}/onnx`);
  const upstream = await huggingFace("api/models/sentence-transformers/all-MiniLM-L6-v2");
  return {
    name: EMBEDDER_NAME,
    description:
      "A sentence-embedding model producing 384-dimensional vectors, in the ONNX int8 export " +
      "chapter 7's retrieval lesson runs with transformers.js.",
    architecture: "BERT (6 layers), mean pooling",
    // The float weights; the total also counts 512 integer position ids, which aren't learned.
    parameterCount: upstream.safetensors?.parameters?.F32 ?? upstream.safetensors?.total ?? 0,
    modelConfig: JSON.parse(readFileSync(new URL("config.json", dir), "utf8")),
    files: [
      hashFile(new URL("onnx/model_quantized.onnx", dir), "application/octet-stream"),
      hashFile(new URL("tokenizer.json", dir), "application/json"),
      hashFile(new URL("config.json", dir), "application/json"),
    ],
    source: {
      repository: `https://huggingface.co/${EMBEDDING_MODEL}`,
      revision: EMBEDDING_REVISION,
      publishedSha256: tree.find(
        (file: { path: string }) => file.path === "onnx/model_quantized.onnx",
      )?.lfs?.oid,
      licence: {
        id: `https://huggingface.co/${EMBEDDING_MODEL}`,
        name: "Apache License 2.0 (as the model card lists it)",
      },
      producer: {
        name: "Sentence Transformers; ONNX export published on Hugging Face by Xenova",
        // ZZ is the "Unknown Region" in Unicode's locale data; the model card doesn't say where.
        country: { countryCode: "ZZ", countryName: "Unknown (not stated by the publisher)" },
      },
    },
  };
}

/** Chapter 7's instruction-tuned GPT-2, from its checkpoint. */
function generatorFacts(): ModelFacts {
  const saved = JSON.parse(readFileSync(new URL("config.json", GENERATOR_CHECKPOINT), "utf8"));
  const shapes: number[][] = saved.shapes;
  return {
    name: GENERATOR_NAME,
    description:
      "GPT-2 124M fine-tuned to follow instructions in chapter 7, lesson 5. It answers the " +
      "retrieval system's questions.",
    architecture: "GPT-2",
    parameterCount: shapes.reduce((sum, shape) => sum + shape.reduce((a, b) => a * b, 1), 0),
    modelConfig: saved.config,
    weights: hashFile(new URL("weights.bin", GENERATOR_CHECKPOINT), "application/octet-stream"),
    config: hashFile(new URL("config.json", GENERATOR_CHECKPOINT), "application/json"),
  };
}

/** Chapter 7's fine-tune, recorded after the fact from lesson 5's settings. */
function generatorRun(code: DerivationFacts["code"]): DerivationFacts {
  const weights = new URL("weights.bin", GENERATOR_CHECKPOINT);
  return {
    derivationId: "gpt2-124m-instruct",
    eventDate: statSync(weights).mtime.toISOString(),
    processType: "Fine-tuning (instruction following, chapter 7)",
    hyperparameters: {
      optimizer: "AdamW (Adam with decoupled weight decay)",
      learningRate: 5e-5,
      weightDecay: 0.1,
      batchSize: 8,
      epochs: 2,
      data: "chapter 7's training split, in batches padded to their longest example",
      record:
        "Recorded after the fact from lesson 5's settings; the run wasn't seeded or logged, so " +
        "its duration is unknown and it can't be reproduced exactly.",
    },
    code,
  };
}

/** The passages a collection passport lists, once the platform verifies it as Book Publisher's. */
async function verifiedListing(
  client: AqClient,
  manifest: Manifest,
  passport: IssuedCredential,
): Promise<ClaimedFile[]> {
  const result = await client.verify("dpp", passport.credentialId);
  const publisher = manifest.authorities.bookPublisher?.did;
  if (!result.verified)
    throw new Error(`the collection passport doesn't verify: ${passport.credentialId}`);
  if (result.issuer !== publisher) {
    throw new Error(`the collection passport is from ${result.issuer}, not Book Publisher`);
  }
  console.log("  the collection passport verifies, and Book Publisher issued it");
  const jwt = await client.fetchCredential(passport.credentialId);
  const payload = JSON.parse(Buffer.from(jwt.split(".")[1] ?? "", "base64url").toString("utf8"));
  return payload.credentialSubject.characteristics.files;
}

/** The parameters in a safetensors file, counted from its header without reading the tensors. */
function safetensorsParameterCount(path: URL): number {
  const fd = openSync(path, "r");
  const length = Buffer.alloc(8);
  readSync(fd, length, 0, 8, 0);
  const header = Buffer.alloc(Number(length.readBigUInt64LE(0)));
  readSync(fd, header, 0, header.length, 8);
  closeSync(fd);
  const tensors: Record<string, { shape?: number[] }> = JSON.parse(header.toString("utf8"));
  return Object.entries(tensors)
    .filter(([name]) => name !== "__metadata__")
    .reduce((sum, [, tensor]) => sum + (tensor.shape ?? []).reduce((a, b) => a * b, 1), 0);
}

// biome-ignore lint/suspicious/noExplicitAny: Hugging Face API responses, read loosely
async function huggingFace(path: string): Promise<any> {
  const response = await fetch(`https://huggingface.co/${path}`);
  if (!response.ok) throw new Error(`Hugging Face ${path}: HTTP ${response.status}`);
  return response.json();
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
