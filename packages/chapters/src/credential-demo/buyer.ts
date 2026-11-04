import { readFileSync } from "node:fs";
import { evaluateLoss } from "@node-llm/core";
import { loadCheckpoint, makeLoaders } from "../ch05-training/shared";
import {
  artifactChecks,
  type ChainClaims,
  claimChecks,
  credentialDigestCheck,
  type DemoCheck,
  decodeJwtPayload,
  evaluationRerunCheck,
  modelLinks,
  readChainClaims,
} from "./checks";
import type { AqClient, VerificationResult } from "./client";
import { REPO_ROOT } from "./env";
import { hashFile } from "./hashing";
import type { Manifest, RunRecord } from "./state";

/**
 * The buyer's side of the demo (Model Deployment Inc.): it holds a model passport URL, the model
 * files and the dataset, and checks them against the credentials.
 */

/** The run with this ID, or the latest run when no ID is given. */
export function pickRun(manifest: Manifest, runId?: string): RunRecord {
  const run = runId ? manifest.runs.find((r) => r.runId === runId) : manifest.runs.at(-1);
  if (!run) {
    throw new Error(runId ? `no run ${runId} in the manifest` : "no runs yet: run pnpm demo:run");
  }
  return run;
}

/** A run's checkpoint name under `checkpoints/`, as `loadCheckpoint` takes it. */
export function checkpointName(run: RunRecord): string {
  return run.checkpoint.replace(/^checkpoints\//, "");
}

/** The batch size of the validation loader; `evaluateLoss` averages per batch, so it matters. */
export const EVALUATION_BATCH_SIZE = 8;

export const VERDICT_PATH = new URL("fixtures/the-verdict.txt", REPO_ROOT);

/** What the buyer learns from the platform and the credentials, before looking at any file. */
export interface ChainReading {
  modelPassportUrl: string;
  platform: VerificationResult;
  claims: ChainClaims;
  credentialCheck: DemoCheck;
}

/** The files the buyer holds: a checkpoint (its name under `checkpoints/`) and the dataset. */
export interface BuyerFiles {
  checkpoint: string;
  dataset: URL;
}

/**
 * Asks the platform to verify the model passport and its whole chain, then reads the model
 * passport, the dataset passport and the evaluation from their did:web URLs.
 */
export async function readChain(client: AqClient, modelPassportUrl: string): Promise<ChainReading> {
  const platform = await client.verify("dpp", modelPassportUrl);
  const modelJwt = await client.fetchCredential(modelPassportUrl);
  const model = decodeJwtPayload(modelJwt);
  const links = modelLinks(model);
  const datasetJwt = await client.fetchCredential(links.datasetPassport.linkURL);
  const evaluationJwt = await client.fetchCredential(links.evaluation.linkURL);

  return {
    modelPassportUrl,
    platform,
    claims: readChainClaims(model, decodeJwtPayload(datasetJwt), decodeJwtPayload(evaluationJwt)),
    credentialCheck: credentialDigestCheck([
      { link: links.datasetPassport, jwt: datasetJwt },
      { link: links.evaluation, jwt: evaluationJwt },
    ]),
  };
}

/** The checks on the buyer's own files: hashes, and a rerun of the evaluation. */
export function fileChecks(reading: ChainReading, files: BuyerFiles): DemoCheck[] {
  const dir = new URL(`checkpoints/${files.checkpoint}/`, REPO_ROOT);
  const digests = {
    weights: hashFile(new URL("weights.bin", dir), "application/octet-stream"),
    config: hashFile(new URL("config.json", dir), "application/json"),
    dataset: hashFile(files.dataset, "text/plain"),
  };
  const measured = rerunEvaluation(files);
  return [
    ...artifactChecks(reading.claims, digests),
    evaluationRerunCheck(reading.claims, measured, harnessVersion()),
  ];
}

/** Every check, in report order. */
export function allChecks(reading: ChainReading, files: BuyerFiles): DemoCheck[] {
  return [reading.credentialCheck, ...claimChecks(reading.claims), ...fileChecks(reading, files)];
}

/** The validation loss of the checkpoint on the dataset's validation split. */
export function rerunEvaluation(files: BuyerFiles): number {
  const gpt = loadCheckpoint(files.checkpoint);
  if (!gpt) throw new Error(`no checkpoint at checkpoints/${files.checkpoint}`);
  const text = readFileSync(files.dataset, "utf8");
  const { val } = makeLoaders(text, gpt.config.contextLength, EVALUATION_BATCH_SIZE);
  const loss = evaluateLoss(gpt, val);
  gpt.dispose();
  return loss;
}

/** The version of the evaluation harness installed here (`@node-llm/core`). */
export function harnessVersion(): string {
  const core = JSON.parse(readFileSync(new URL("packages/core/package.json", REPO_ROOT), "utf8"));
  return core.version;
}

// ============================================================================
// Reports
// ============================================================================

/**
 * Each credential's own `name`, for every credential in the platform's result. A credential that
 * can't be read keeps no name, and the tree falls back to the name of the link that reached it.
 */
export async function credentialNames(
  client: AqClient,
  result: VerificationResult,
): Promise<Map<string, string>> {
  const names = new Map<string, string>();
  for (const credentialId of credentialIds(result)) {
    try {
      const credential = decodeJwtPayload(await client.fetchCredential(credentialId));
      const name = credential.name ?? credential.credentialSubject?.name;
      if (typeof name === "string") names.set(credentialId, name);
    } catch {
      // Unreadable credentials show up as failures in the tree anyway.
    }
  }
  return names;
}

function credentialIds(result: VerificationResult, ids = new Set<string>()): Set<string> {
  ids.add(result.credentialId);
  for (const child of result.upstreamResults ?? []) credentialIds(child, ids);
  return ids;
}

/**
 * The platform's result as a tree, one line per credential, with any check that didn't pass. A
 * credential linked from several places is shown in full once, and marked where it repeats.
 */
export function printPlatformTree(
  result: VerificationResult,
  names: Map<string, string>,
  indent = "",
  shown = new Set<string>(),
): void {
  const name = names.get(result.credentialId) ?? result.link?.linkName ?? result.credentialId;
  const status = result.verified ? "VERIFIED" : "NOT VERIFIED";
  if (shown.has(result.credentialId)) {
    console.log(`${indent}${status.padEnd(12)} ${result.credentialType}  ${name}  (shown above)`);
    return;
  }
  shown.add(result.credentialId);
  const notPassed = Object.entries(result.checks)
    .filter(([, check]) => check && check.status !== "pass")
    .map(([key, check]) => `${key} ${check?.status}`);
  const passed = Object.values(result.checks).filter((check) => check?.status === "pass").length;
  const summary = [`${passed} checks pass`, ...notPassed].join(", ");
  console.log(`${indent}${status.padEnd(12)} ${result.credentialType}  ${name}  (${summary})`);
  for (const error of result.errors)
    console.log(`${indent}  error ${error.code}: ${error.message}`);
  for (const child of result.upstreamResults ?? []) {
    printPlatformTree(child, names, `${indent}  `, shown);
  }
}

export function printChecks(checks: DemoCheck[]): void {
  for (const check of checks) {
    console.log(`  ${check.passed ? "pass" : "FAIL"}  ${check.group.padEnd(11)} ${check.name}`);
    console.log(`        ${check.detail}`);
  }
}
