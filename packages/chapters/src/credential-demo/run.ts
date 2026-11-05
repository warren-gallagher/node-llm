import { availableParallelism, cpus } from "node:os";
import { evaluateLoss, GPT_CONFIG_124M, GptModel, trainModel } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";
import { loadCheckpoint, loadVerdict, makeLoaders, saveCheckpoint } from "../ch05-training/shared";
import {
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
import { AqClient } from "./client";
import { loadDemoEnv, REPO_ROOT } from "./env";
import { hashFile } from "./hashing";
import {
  authorityId,
  datasetFacts,
  gitCode,
  measureEvaluation,
  printVerification,
} from "./pipeline";
import { PRNG_NAME, seedMathRandom } from "./seeded";
import {
  type IssuedCredential,
  loadManifest,
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
  const evaluation = measureEvaluation(gpt, code);
  gpt.dispose();
  return evaluation;
}

function libtensorflowVersion(): string {
  const backend = tf.backend() as unknown as { binding?: { TF_Version?: string } };
  return backend.binding?.TF_Version ?? "unknown";
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
