import { readFileSync } from "node:fs";
import { checkpointName, pickRun } from "./buyer";
import {
  type DemoContext,
  type DerivationFacts,
  derivationDte,
  evaluationDcc,
  type ModelFacts,
  modelDpp,
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
import { loadQuantized, QUANTIZATION, QUANTIZED_WEIGHTS, quantizeCheckpoint } from "./quantize";
import {
  type DerivationRecord,
  type IssuedCredential,
  loadManifest,
  partyIdentities,
  type RunRecord,
  saveManifest,
  toIssued,
} from "./state";

/**
 * Credential demo, phase 2: the first derivation. Model Publisher quantizes a run's model to
 * int8, Model Auditor evaluates the quantized weights (the parent's evaluation doesn't carry
 * over), and Model Publisher issues the derivation event and the quantized model's passport. The
 * passport carries the parent's training-rights claim forward, and replaces its loss claim with
 * the new evaluation.
 *
 * Usage: pnpm demo:quantize [runId] [--dry-run]   (default: the latest run)
 *
 * `--dry-run` has the platform validate and sign each credential without publishing it, and
 * leaves the manifest unchanged.
 */

const PROCESS_TYPE = "Quantization (int8)";

async function main(): Promise<void> {
  const env = loadDemoEnv();
  const client = new AqClient(env);
  const manifest = loadManifest(env.apiUrl, env.tenantId);
  const args = process.argv.slice(2);
  const dryRun = args.includes("--dry-run");
  const parentRun = pickRun(
    manifest,
    args.find((arg) => !arg.startsWith("--")),
  );
  const store = { store: !dryRun };
  const { modelAuditor } = manifest.accreditations;
  if (!modelAuditor || !manifest.dataset) {
    throw new Error("setup or the first run is missing: run pnpm demo:setup and pnpm demo:run");
  }
  const ctx: DemoContext = {
    apiUrl: env.apiUrl,
    parties: partyIdentities(manifest),
    date: new Date().toISOString().slice(0, 10),
  };
  const code = gitCode();
  const dataset = datasetFacts();
  const parent = parentFacts(parentRun);

  // 1. Quantize, as Model Publisher.
  const derivationId = `${new Date().toISOString().slice(0, 19).replaceAll(":", "-")}Z-int8`;
  const checkpoint = `verdict-demo/derived/${dryRun ? "dry-run" : derivationId}`;
  if (dryRun) console.log("Dry run: the platform validates and signs, but publishes nothing");
  console.log(`Derivation ID: ${derivationId}`);
  console.log(`Quantizing ${parent.name}`);
  const eventDate = new Date().toISOString();
  const started = performance.now();
  quantizeCheckpoint(checkpointName(parentRun), checkpoint);
  const derivation: DerivationFacts = {
    derivationId,
    eventDate,
    processType: PROCESS_TYPE,
    hyperparameters: QUANTIZATION,
    code,
    durationSeconds: Math.round((performance.now() - started) / 100) / 10,
  };
  const derived = derivedFacts(parent, checkpoint);
  console.log(`  weights sha256 ${derived.weights.sha256}`);

  // 2. Evaluate the quantized weights, as Model Auditor.
  const gpt = loadQuantized(checkpoint);
  const evaluation = measureEvaluation(gpt, code);
  gpt.dispose();
  console.log(`  (the parent's was ${parentRun.validationLoss})`);

  // 3. The credentials.
  const event = toIssued(
    await client.sign(
      "dte",
      authorityId(manifest, "modelPublisher"),
      derivationDte(ctx, parent, derived, derivation, { parentPassport: parentRun.passport }),
      store,
    ),
  );
  console.log(`\n  derivation DTE (Model Publisher): ${event.credentialId}`);
  const evaluationCredential = toIssued(
    await client.sign(
      "dcc",
      authorityId(manifest, "modelAuditor"),
      evaluationDcc(ctx, dataset, derived, evaluation, modelAuditor),
      store,
    ),
  );
  console.log(`  evaluation DCC (Model Auditor): ${evaluationCredential.credentialId}`);
  const passport = toIssued(
    await client.sign(
      "dpp",
      authorityId(manifest, "modelPublisher"),
      modelDpp(ctx, dataset, derived, evaluation, {
        datasetPassport: manifest.dataset.passport,
        rights: manifest.dataset.rights,
        trainingEvent: event,
        evaluation: evaluationCredential,
        derivedFrom: {
          parentName: parent.name,
          parentPassport: parentRun.passport,
          processType: PROCESS_TYPE,
        },
      }),
      store,
    ),
  );
  console.log(`  model passport (Model Publisher): ${passport.credentialId}`);

  if (dryRun) {
    console.log("\nDry run complete: all three credentials passed the platform's validation.");
    return;
  }

  const record: DerivationRecord = {
    derivationId,
    kind: "quantization",
    parentRunId: parentRun.runId,
    checkpoint: `checkpoints/${checkpoint}`,
    weightsFile: QUANTIZED_WEIGHTS,
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

/** The parent model as its passport described it, rebuilt from its checkpoint. */
function parentFacts(run: RunRecord): ModelFacts {
  const dir = new URL(`${run.checkpoint}/`, REPO_ROOT);
  const weights = hashFile(new URL("weights.bin", dir), "application/octet-stream");
  if (weights.sha256 !== run.weightsSha256) {
    throw new Error(`${run.checkpoint}/weights.bin isn't the model run ${run.runId} trained`);
  }
  const saved = JSON.parse(readFileSync(new URL("config.json", dir), "utf8"));
  return {
    name: `Verdict GPT ${run.runId}`,
    description: "The parent model.",
    architecture: "GPT-2",
    parameterCount: parameterCount(saved.shapes),
    modelConfig: saved.config,
    weights,
    config: hashFile(new URL("config.json", dir), "application/json"),
  };
}

/** The quantized model: the parent's architecture and size, in new files. */
function derivedFacts(parent: ModelFacts, checkpoint: string): ModelFacts {
  const dir = new URL(`checkpoints/${checkpoint}/`, REPO_ROOT);
  return {
    name: `${parent.name} (int8)`,
    description:
      `${parent.name}, quantized to int8 with one scale per tensor. ` +
      "Anyone holding the parent's weights can rerun the quantization and compare hashes.",
    architecture: "GPT-2 (int8 weights)",
    parameterCount: parent.parameterCount,
    modelConfig: parent.modelConfig,
    weights: hashFile(new URL(QUANTIZED_WEIGHTS, dir), "application/octet-stream"),
    config: hashFile(new URL("config.json", dir), "application/json"),
  };
}

function parameterCount(shapes: number[][]): number {
  return shapes.reduce((sum, shape) => sum + shape.reduce((a, b) => a * b, 1), 0);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
