import { evaluateLoss, formatInput, formatResponse, trainModel } from "@node-llm/core";
import { loadCheckpoint, makeLoaders, saveCheckpoint } from "../ch05-training/shared";
import { loadInstructionSplits } from "../ch07-instruction/shared";
import { checkpointName, pickRun } from "./buyer";
import {
  type DatasetFacts,
  type DemoContext,
  type DerivationFacts,
  datasetDpp,
  derivationDte,
  evaluationDcc,
  type ModelFacts,
  modelDpp,
  rightsDcc,
} from "./claims";
import { AqClient } from "./client";
import { loadDemoEnv, REPO_ROOT } from "./env";
import { hashFile } from "./hashing";
import {
  authorityId,
  datasetFacts,
  gitCode,
  instructionFacts,
  measureEvaluation,
  parentModelFacts,
  printVerification,
} from "./pipeline";
import { PRNG_NAME, seedMathRandom } from "./seeded";
import {
  type DatasetRecord,
  type DerivationRecord,
  type IssuedCredential,
  loadManifest,
  type Manifest,
  partyIdentities,
  saveManifest,
  toIssued,
} from "./state";

/**
 * Credential demo, phase 2: the second derivation. Model Publisher fine-tunes a run's model on
 * chapter 7's instruction data, which first gets its own rights DCC (Book Auditor) and dataset
 * passport (Book Publisher). Model Auditor evaluates the fine-tuned weights on The Verdict's
 * validation split, as for every model. The fine-tuned passport carries The Verdict's rights
 * claim forward, adds the instruction data's, and replaces the loss claim with the new
 * evaluation.
 *
 * Unlike quantization, fine-tuning can't be rerun bit for bit, so a buyer can only check that
 * the weight changes look like a fine-tune.
 *
 * Usage: pnpm demo:finetune [runId] [--dry-run]   (default: the latest run)
 *
 * `--dry-run` has the platform validate and sign each credential without publishing it, and
 * leaves the manifest unchanged.
 */

const PROCESS_TYPE = "Fine-tuning (next-token prediction on instruction data)";
const SEED = 42;
const EPOCHS = 1;
const BATCH_SIZE = 8;
const LEARNING_RATE = 5e-5;
const WEIGHT_DECAY = 0.1;

async function main(): Promise<void> {
  const env = loadDemoEnv();
  const client = new AqClient(env);
  const manifest = loadManifest(env.apiUrl, env.tenantId);
  const args = process.argv.slice(2);
  const dryRun = args.includes("--dry-run");
  const store = { store: !dryRun };
  const parentRun = pickRun(
    manifest,
    args.find((arg) => !arg.startsWith("--")),
  );
  const { bookAuditor, modelAuditor } = manifest.accreditations;
  if (!bookAuditor || !modelAuditor || !manifest.dataset) {
    throw new Error("setup or the first run is missing: run pnpm demo:setup and pnpm demo:run");
  }
  const ctx: DemoContext = {
    apiUrl: env.apiUrl,
    parties: partyIdentities(manifest),
    date: new Date().toISOString().slice(0, 10),
  };
  const code = gitCode();
  const verdict = datasetFacts();
  const parent = parentModelFacts(parentRun);
  const derivationId = `${new Date().toISOString().slice(0, 19).replaceAll(":", "-")}Z-ft`;
  console.log(`Derivation ID: ${derivationId}`);
  if (dryRun) console.log("Dry run: the platform validates and signs, but publishes nothing");

  // 1. The fine-tuning data's credentials, reused while the file is unchanged.
  const { train } = await loadInstructionSplits();
  const instructions = await instructionFacts(verdict);
  const data = await dataCredentials(manifest, client, ctx, instructions, bookAuditor, store);

  // 2. Fine-tune, as Model Publisher.
  const checkpoint = `verdict-demo/derived/${dryRun ? "dry-run-ft" : derivationId}`;
  const text = train.map((entry) => formatInput(entry) + formatResponse(entry)).join("\n\n");
  console.log(`\nFine-tuning ${parent.name} on ${train.length} instruction examples`);
  const derivation = fineTune(checkpointName(parentRun), checkpoint, text, derivationId, code);
  const derived = derivedFacts(parent, checkpoint);
  console.log(`  weights sha256 ${derived.weights.sha256}`);

  // 3. Evaluate, as Model Auditor, on The Verdict's validation split like every model.
  const gpt = loadCheckpoint(checkpoint);
  if (!gpt) throw new Error(`no checkpoint at ${checkpoint}`);
  const evaluation = measureEvaluation(gpt, code);
  gpt.dispose();
  console.log(`  (the parent's was ${parentRun.validationLoss})`);

  // 4. The credentials.
  const event = toIssued(
    await client.sign(
      "dte",
      authorityId(manifest, "modelPublisher"),
      derivationDte(ctx, parent, derived, derivation, {
        parentPassport: parentRun.passport,
        data: { dataset: instructions, passport: data.passport, rights: data.rights },
      }),
      store,
    ),
  );
  console.log(`\n  derivation DTE (Model Publisher): ${event.credentialId}`);
  const evaluationCredential = toIssued(
    await client.sign(
      "dcc",
      authorityId(manifest, "modelAuditor"),
      evaluationDcc(ctx, verdict, derived, evaluation, modelAuditor),
      store,
    ),
  );
  console.log(`  evaluation DCC (Model Auditor): ${evaluationCredential.credentialId}`);
  const passport = toIssued(
    await client.sign(
      "dpp",
      authorityId(manifest, "modelPublisher"),
      modelDpp(ctx, verdict, derived, evaluation, {
        datasetPassport: manifest.dataset.passport,
        rights: manifest.dataset.rights,
        trainingEvent: event,
        evaluation: evaluationCredential,
        derivedFrom: {
          parentName: parent.name,
          parentPassport: parentRun.passport,
          processType: "Fine-tuning",
          data: { dataset: instructions, rights: data.rights },
        },
      }),
      store,
    ),
  );
  console.log(`  model passport (Model Publisher): ${passport.credentialId}`);

  if (dryRun) {
    console.log("\nDry run complete: every credential passed the platform's validation.");
    return;
  }

  const record: DerivationRecord = {
    derivationId,
    kind: "fine-tuning",
    parentRunId: parentRun.runId,
    checkpoint: `checkpoints/${checkpoint}`,
    weightsFile: "weights.bin",
    weightsSha256: derived.weights.sha256,
    configSha256: derived.config.sha256,
    validationLoss: evaluation.validationLoss,
    event,
    evaluation: evaluationCredential,
    passport,
  };
  manifest.derivations = [...(manifest.derivations ?? []), record];
  saveManifest(manifest);

  console.log("\nPlatform verification of each credential issued:");
  const issued: IssuedCredential[] = [event, evaluationCredential, passport];
  for (const credential of issued) {
    printVerification(await client.verify(credential.type, credential.credentialId));
  }
  console.log(`\nCheck it as a buyer: pnpm demo:verify ${derivationId}`);
}

/**
 * The fine-tuning data's rights DCC and passport: the ones in the manifest while the file's hash
 * is unchanged, or new ones.
 */
async function dataCredentials(
  manifest: Manifest,
  client: AqClient,
  ctx: DemoContext,
  dataset: DatasetFacts,
  accreditation: IssuedCredential,
  store: { store: boolean },
): Promise<DatasetRecord> {
  console.log(`\nFine-tuning data: ${dataset.title}, sha256 ${dataset.file.sha256}`);
  if (manifest.fineTuningDataset?.sha256 === dataset.file.sha256) {
    console.log("  reusing its rights DCC and passport");
    return manifest.fineTuningDataset;
  }
  const rights = toIssued(
    await client.sign(
      "dcc",
      authorityId(manifest, "bookAuditor"),
      rightsDcc(ctx, dataset, accreditation),
      store,
    ),
  );
  console.log(`  rights DCC (Book Auditor): ${rights.credentialId}`);
  const passport = toIssued(
    await client.sign(
      "dpp",
      authorityId(manifest, "bookPublisher"),
      datasetDpp(ctx, dataset, rights),
      store,
    ),
  );
  console.log(`  dataset passport (Book Publisher): ${passport.credentialId}`);
  const record = { sha256: dataset.file.sha256, rights, passport };
  if (store.store) {
    manifest.fineTuningDataset = record;
    saveManifest(manifest);
  }
  return record;
}

/** Fine-tunes the parent checkpoint with a seeded `Math.random`, and saves the result. */
function fineTune(
  parent: string,
  checkpoint: string,
  text: string,
  derivationId: string,
  code: DerivationFacts["code"],
): DerivationFacts {
  const eventDate = new Date().toISOString();
  const started = performance.now();
  const restore = seedMathRandom(SEED);
  const gpt = loadCheckpoint(parent);
  if (!gpt) throw new Error(`no checkpoint at ${parent}`);
  const { train, val } = makeLoaders(text, gpt.config.contextLength, BATCH_SIZE);
  trainModel(gpt, train, {
    epochs: EPOCHS,
    learningRate: LEARNING_RATE,
    weightDecay: WEIGHT_DECAY,
    evalEvery: 20,
    valLoader: val,
    onEval: (p) =>
      console.log(`  step ${p.step}: train ${p.trainLoss.toFixed(3)}, val ${p.valLoss.toFixed(3)}`),
  });
  const finalLoss = { train: evaluateLoss(gpt, train), validation: evaluateLoss(gpt, val) };
  restore();
  saveCheckpoint(gpt, checkpoint);
  gpt.dispose();
  return {
    derivationId,
    eventDate,
    processType: PROCESS_TYPE,
    hyperparameters: {
      optimizer: "AdamW (Adam with decoupled weight decay)",
      learningRate: LEARNING_RATE,
      weightDecay: WEIGHT_DECAY,
      batchSize: BATCH_SIZE,
      epochs: EPOCHS,
      seed: SEED,
      prng: PRNG_NAME,
      data:
        "chapter 7's training split (the first 85% of the examples), each formatted with the " +
        "book's prompt template, joined, and cut into 64-token windows; the last 10% of the " +
        "text held out for validation",
      finalLoss,
    },
    code,
    durationSeconds: Math.round((performance.now() - started) / 100) / 10,
  };
}

/** The fine-tuned model: the parent's architecture and size, in new files. */
function derivedFacts(parent: ModelFacts, checkpoint: string): ModelFacts {
  const dir = new URL(`checkpoints/${checkpoint}/`, REPO_ROOT);
  return {
    name: `${parent.name} (fine-tuned)`,
    description:
      `${parent.name}, fine-tuned for one epoch on chapter 7's instruction data. ` +
      "Fine-tuning can't be rerun bit for bit; a buyer can compare its weights with the parent's.",
    architecture: "GPT-2",
    parameterCount: parent.parameterCount,
    modelConfig: parent.modelConfig,
    weights: hashFile(new URL("weights.bin", dir), "application/octet-stream"),
    config: hashFile(new URL("config.json", dir), "application/json"),
  };
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
