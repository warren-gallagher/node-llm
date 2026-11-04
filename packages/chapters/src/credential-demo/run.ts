import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { availableParallelism, cpus } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { evaluateLoss, GPT_CONFIG_124M, GptModel, trainModel } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";
import { encode } from "gpt-tokenizer/encoding/r50k_base";
import { loadCheckpoint, loadVerdict, makeLoaders, saveCheckpoint } from "../ch05-training/shared";
import {
  type DatasetFacts,
  type DemoContext,
  datasetDpp,
  type EvaluationFacts,
  evaluationDcc,
  type ModelFacts,
  modelDpp,
  rightsDcc,
  type TrainingFacts,
  trainingDte,
} from "./claims";
import { AqClient, type VerificationResult } from "./client";
import { loadDemoEnv, REPO_ROOT } from "./env";
import { digestMultibase, hashFile, sha256Hex } from "./hashing";
import type { PartyKey } from "./parties";
import { PRNG_NAME, seedMathRandom } from "./seeded";
import {
  type IssuedCredential,
  loadManifest,
  type Manifest,
  partyIdentities,
  saveManifest,
  toIssued,
} from "./state";

/**
 * Credential demo, one pipeline run. Each party's step issues its credential through the platform:
 *
 * 1. Book Auditor and Book Publisher: the rights DCC and the dataset passport for The Verdict
 *    (reused while the file's hash is unchanged);
 * 2. Model Publisher: a seeded training run (chapter 5's small configuration), then the DTE;
 * 3. Model Auditor: hashes the saved weights, measures the validation loss, issues the evaluation;
 * 4. Model Publisher: the model passport, linking the whole chain.
 *
 * It ends by asking the platform to verify each credential issued. Run `pnpm demo:setup` first.
 */

const SEED = 42;
const EPOCHS = 10;
const BATCH_SIZE = 8;
const LEARNING_RATE = 0.001;
const WEIGHT_DECAY = 0.1;
const MODEL_CONFIG = {
  ...GPT_CONFIG_124M,
  contextLength: 64,
  embDim: 128,
  numHeads: 4,
  numLayers: 2,
};

const VERDICT_SOURCE_URL =
  "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";
const LOSS_METRIC = "cross-entropy loss (nats per token)";
const VALIDATION_SPLIT = `validation: the last 10% of the text's characters, in ${MODEL_CONFIG.contextLength}-token windows`;

async function main(): Promise<void> {
  const env = loadDemoEnv();
  const client = new AqClient(env);
  const manifest = loadManifest(env.apiUrl, env.tenantId);
  const { bookAuditor, modelAuditor } = manifest.accreditations;
  if (!bookAuditor || !modelAuditor) {
    throw new Error("the evaluators aren't accredited yet: run pnpm demo:setup");
  }
  const ctx: DemoContext = {
    apiUrl: env.apiUrl,
    parties: partyIdentities(manifest),
    date: new Date().toISOString().slice(0, 10),
  };
  const code = gitCode();
  const issued: IssuedCredential[] = [];

  // 1. The dataset's credentials.
  const dataset = datasetFacts();
  console.log(`Dataset: ${dataset.title}, sha256 ${dataset.file.sha256}`);
  if (manifest.dataset?.sha256 === dataset.file.sha256) {
    console.log(`  reusing its rights DCC and passport`);
  } else {
    const rights = toIssued(
      await client.sign(
        "dcc",
        authorityId(manifest, "bookAuditor"),
        rightsDcc(ctx, dataset, bookAuditor),
      ),
    );
    console.log(`  rights DCC (Book Auditor): ${rights.credentialId}`);
    const passport = toIssued(
      await client.sign(
        "dpp",
        authorityId(manifest, "bookPublisher"),
        datasetDpp(ctx, dataset, rights),
      ),
    );
    console.log(`  dataset passport (Book Publisher): ${passport.credentialId}`);
    manifest.dataset = { sha256: dataset.file.sha256, rights, passport };
    saveManifest(manifest);
    issued.push(rights, passport);
  }
  const { rights, passport: datasetPassport } = manifest.dataset;

  // 2. Training, as Model Publisher.
  const runId = `${new Date().toISOString().slice(0, 19).replaceAll(":", "-")}Z`;
  const { model, training, checkpoint } = train(runId, code);
  const trainingEvent = toIssued(
    await client.sign(
      "dte",
      authorityId(manifest, "modelPublisher"),
      trainingDte(ctx, dataset, model, training, { datasetPassport, rights }),
    ),
  );
  console.log(`  training DTE (Model Publisher): ${trainingEvent.credentialId}`);

  // 3. Evaluation, as Model Auditor, of the files on disk.
  const evaluation = evaluate(checkpoint, model, code);
  const evaluationCredential = toIssued(
    await client.sign(
      "dcc",
      authorityId(manifest, "modelAuditor"),
      evaluationDcc(ctx, dataset, model, evaluation, modelAuditor),
    ),
  );
  console.log(`  evaluation DCC (Model Auditor): ${evaluationCredential.credentialId}`);

  // 4. The model passport.
  const modelPassport = toIssued(
    await client.sign(
      "dpp",
      authorityId(manifest, "modelPublisher"),
      modelDpp(ctx, dataset, model, evaluation, {
        datasetPassport,
        rights,
        trainingEvent,
        evaluation: evaluationCredential,
      }),
    ),
  );
  console.log(`  model passport (Model Publisher): ${modelPassport.credentialId}`);

  manifest.runs.push({
    runId,
    checkpoint: `checkpoints/${checkpoint}`,
    weightsSha256: model.weights.sha256,
    configSha256: model.config.sha256,
    validationLoss: evaluation.validationLoss,
    trainingEvent,
    evaluation: evaluationCredential,
    passport: modelPassport,
  });
  saveManifest(manifest);
  issued.push(trainingEvent, evaluationCredential, modelPassport);

  console.log("\nPlatform verification of each credential issued:");
  for (const credential of issued) {
    printVerification(await client.verify(credential.type, credential.credentialId));
  }
}

function authorityId(manifest: Manifest, key: PartyKey): string {
  const record = manifest.authorities[key];
  if (!record) throw new Error(`no authority for ${key}: run pnpm demo:setup`);
  return record.id;
}

function datasetFacts(): DatasetFacts {
  const path = new URL("fixtures/the-verdict.txt", REPO_ROOT);
  const tokenizerDir = packageDir("gpt-tokenizer");
  const tokenizerPackage = JSON.parse(readFileSync(join(tokenizerDir, "package.json"), "utf8"));
  const encoding = readFileSync(join(tokenizerDir, "data", "r50k_base.tiktoken"));

  return {
    title: "The Verdict (Edith Wharton, 1908)",
    description:
      "Edith Wharton's short story, in plain text: the training corpus of chapter 5 of " +
      "Raschka's Build a Large Language Model (From Scratch).",
    file: hashFile(path, "text/plain"),
    sourceUrl: VERDICT_SOURCE_URL,
    tokenCount: encode(loadVerdict()).length,
    tokenizer: {
      name: "r50k_base",
      package: "gpt-tokenizer",
      version: tokenizerPackage.version,
      digestMultibase: digestMultibase(sha256Hex(encoding)),
    },
  };
}

/** Trains the model with a seeded `Math.random`, saves it, and hashes the saved files. */
function train(
  runId: string,
  code: TrainingFacts["code"],
): { model: ModelFacts; training: TrainingFacts; checkpoint: string } {
  console.log(`\nTraining run ${runId} (seed ${SEED})`);
  const eventDate = new Date().toISOString();
  const started = performance.now();
  const restore = seedMathRandom(SEED);

  const { train: trainLoader, val } = makeLoaders(
    loadVerdict(),
    MODEL_CONFIG.contextLength,
    BATCH_SIZE,
  );
  const gpt = new GptModel(MODEL_CONFIG);
  trainModel(gpt, trainLoader, {
    epochs: EPOCHS,
    learningRate: LEARNING_RATE,
    weightDecay: WEIGHT_DECAY,
    // Evaluation shuffles the training loader, which draws on the seeded generator, so this
    // setting changes the weights. It matches ch05-training/train.ts.
    evalEvery: 10,
    valLoader: val,
    onEval: (p) =>
      console.log(`  step ${p.step}: train ${p.trainLoss.toFixed(3)}, val ${p.valLoss.toFixed(3)}`),
  });
  const finalLoss = { train: evaluateLoss(gpt, trainLoader), validation: evaluateLoss(gpt, val) };
  restore();
  const durationSeconds = Math.round((performance.now() - started) / 100) / 10;

  const checkpoint = `verdict-demo/runs/${runId}`;
  saveCheckpoint(gpt, checkpoint);
  const parameterCount = gpt.numParameters();
  gpt.dispose();
  const dir = new URL(`checkpoints/${checkpoint}/`, REPO_ROOT);
  const weights = hashFile(new URL("weights.bin", dir), "application/octet-stream");
  const config = hashFile(new URL("config.json", dir), "application/json");
  console.log(
    `  final loss: train ${finalLoss.train.toFixed(3)}, val ${finalLoss.validation.toFixed(3)}`,
  );
  console.log(`  ${parameterCount.toLocaleString()} parameters, weights sha256 ${weights.sha256}`);

  return {
    checkpoint,
    model: {
      name: `Verdict GPT ${runId}`,
      description:
        `A ${MODEL_CONFIG.numLayers}-layer, ${MODEL_CONFIG.embDim}-dimension GPT-2-style model ` +
        "pretrained on The Verdict by the credential demo.",
      architecture: "GPT-2",
      parameterCount,
      modelConfig: MODEL_CONFIG,
      weights,
      config,
    },
    training: {
      runId,
      eventDate,
      durationSeconds,
      epochs: EPOCHS,
      hyperparameters: {
        optimizer: "AdamW (Adam with decoupled weight decay)",
        learningRate: LEARNING_RATE,
        weightDecay: WEIGHT_DECAY,
        batchSize: BATCH_SIZE,
      },
      split: { train: 0.9, validation: 0.1 },
      finalLoss,
      code,
      reproduction: {
        seed: SEED,
        prng: PRNG_NAME,
        threads: Number(process.env.TF_NUM_INTRAOP_THREADS) || availableParallelism(),
        runtime: {
          node: process.version,
          "tfjs-node": tf.version["tfjs-node"],
          libtensorflow: libtensorflowVersion(),
        },
        cpu: cpus()[0]?.model ?? "unknown",
      },
    },
  };
}

/**
 * The Model Auditor's step: it loads the weights from disk, checks they are the files the builder
 * named, and measures the validation loss with the shared harness.
 */
function evaluate(
  checkpoint: string,
  model: ModelFacts,
  code: TrainingFacts["code"],
): EvaluationFacts {
  const dir = new URL(`checkpoints/${checkpoint}/`, REPO_ROOT);
  const weights = hashFile(new URL("weights.bin", dir), "application/octet-stream");
  if (weights.sha256 !== model.weights.sha256) {
    throw new Error(`${checkpoint}/weights.bin changed after training`);
  }
  const gpt = loadCheckpoint(checkpoint);
  if (!gpt) throw new Error(`no checkpoint at ${checkpoint}`);
  const { val } = makeLoaders(loadVerdict(), gpt.config.contextLength, BATCH_SIZE);
  const validationLoss = evaluateLoss(gpt, val);
  gpt.dispose();
  console.log(`\nEvaluation: validation loss ${validationLoss}`);

  const corePackage = JSON.parse(
    readFileSync(new URL("packages/core/package.json", REPO_ROOT), "utf8"),
  );
  return {
    validationLoss,
    harness: { name: "@node-llm/core evaluateLoss", version: corePackage.version, ...code },
    split: VALIDATION_SPLIT,
    metric: LOSS_METRIC,
  };
}

/** The repository (as an https URL) and commit, warning if the working tree has changes. */
function gitCode(): TrainingFacts["code"] {
  const remote = git("remote", "get-url", "origin");
  const ssh = /^git@([^:]+):(.+?)(?:\.git)?$/.exec(remote);
  const repository = ssh ? `https://${ssh[1]}/${ssh[2]}` : remote.replace(/\.git$/, "");
  if (git("status", "--porcelain") !== "") {
    console.warn("warning: the working tree has uncommitted changes, so the recorded commit");
    console.warn("doesn't fully describe the code that ran\n");
  }
  return { repository, commit: git("rev-parse", "HEAD") };
}

function git(...args: string[]): string {
  return execFileSync("git", args, { cwd: fileURLToPath(REPO_ROOT), encoding: "utf8" }).trim();
}

/** The installed package's directory (its `exports` don't include package.json or data files). */
function packageDir(name: string): string {
  let dir = dirname(createRequire(import.meta.url).resolve(name));
  while (!dir.endsWith(`node_modules/${name}`)) {
    const parent = dirname(dir);
    if (parent === dir) throw new Error(`can't find the ${name} package directory`);
    dir = parent;
  }
  return dir;
}

function libtensorflowVersion(): string {
  const backend = tf.backend() as unknown as { binding?: { TF_Version?: string } };
  return backend.binding?.TF_Version ?? "unknown";
}

function printVerification(result: VerificationResult): void {
  console.log(
    `\n${result.verified ? "VERIFIED" : "NOT VERIFIED"}  ${result.credentialType}  ${result.credentialId}`,
  );
  for (const [name, check] of Object.entries(result.checks)) {
    if (check) console.log(`  ${check.status.padEnd(7)} ${name}: ${check.message}`);
  }
  for (const error of result.errors) console.log(`  error   ${error.code}: ${error.message}`);
  for (const warning of result.warnings)
    console.log(`  warning ${warning.code}: ${warning.message}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
