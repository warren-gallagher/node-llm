import { readFileSync } from "node:fs";
import { isDeepStrictEqual } from "node:util";
import { evaluateLoss } from "@node-llm/core";
import { loadCheckpoint, makeLoaders } from "../ch05-training/shared";
import {
  artifactChecks,
  assessedProducts,
  type ChainClaims,
  type ClassifiedClaim,
  claimChecks,
  classifyClaims,
  credentialDigestCheck,
  type DemoCheck,
  decodeJwtPayload,
  evaluationRerunCheck,
  inheritanceCheck,
  type Link,
  modelLinks,
  quantizationRerunChecks,
  readChainClaims,
  weightComparisonCheck,
} from "./checks";
import type { AqClient, VerificationResult } from "./client";
import { REPO_ROOT } from "./env";
import { weightChangeProfile } from "./forensics";
import { hashFile, sha256Hex } from "./hashing";
import {
  dequantizeWeights,
  isQuantized,
  loadQuantized,
  QUANTIZATION,
  QUANTIZED_WEIGHTS,
  quantizeWeights,
} from "./quantize";
import type { IssuedCredential, Manifest, RunRecord } from "./state";

/**
 * The buyer's side of the demo (Model Deployment Inc.): it holds a model passport URL, the model
 * files and the dataset, and checks them against the credentials. The model is a run's, or one
 * derived from a run's model; for a derived model the buyer also holds the parent's weights.
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
export function checkpointName(run: { checkpoint: string }): string {
  return run.checkpoint.replace(/^checkpoints\//, "");
}

/** A model the buyer can check: its passport, and the files the buyer holds for it. */
export interface BuyerModel {
  id: string;
  /** How the model was made: trained by a run, or derived from a run's model. */
  kind: "run" | "quantization" | "fine-tuning";
  passport: IssuedCredential;
  files: BuyerFiles;
}

/**
 * The run or derivation with this ID, or the latest run when no ID is given. Derived models
 * come with their parent's checkpoint, which the derivation check needs.
 */
export function pickModel(manifest: Manifest, id?: string): BuyerModel {
  const derivation = id ? manifest.derivations?.find((d) => d.derivationId === id) : undefined;
  if (derivation) {
    const parent = pickRun(manifest, derivation.parentRunId);
    return {
      id: derivation.derivationId,
      kind: derivation.kind,
      passport: derivation.passport,
      files: {
        checkpoint: checkpointName(derivation),
        weightsFile: derivation.weightsFile,
        dataset: VERDICT_PATH,
        parentCheckpoint: checkpointName(parent),
      },
    };
  }
  const run = pickRun(manifest, id);
  return {
    id: run.runId,
    kind: "run",
    passport: run.passport,
    files: { checkpoint: checkpointName(run), weightsFile: "weights.bin", dataset: VERDICT_PATH },
  };
}

/** Every model in the manifest, runs and derivations, newest first. */
export function listModels(manifest: Manifest): { id: string; modelPassport: string }[] {
  const models = [
    ...manifest.runs.map((run) => ({ id: run.runId, modelPassport: run.passport.credentialId })),
    ...(manifest.derivations ?? []).map((d) => ({
      id: d.derivationId,
      modelPassport: d.passport.credentialId,
    })),
  ];
  return models.sort((a, b) => b.id.localeCompare(a.id));
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
  /** For a derived model: its passport's claims classified against the parent's. */
  inheritance?: Inheritance;
}

/** A derived model's claims against its parent's, and what each cited DCC assessed. */
export interface Inheritance {
  // biome-ignore lint/suspicious/noExplicitAny: credentials are read loosely, like any JSON
  modelPassport: any;
  // biome-ignore lint/suspicious/noExplicitAny: credentials are read loosely, like any JSON
  parentPassport: any;
  claims: ClassifiedClaim[];
  assessed: Record<string, string[]>;
}

/**
 * The files the buyer holds: a checkpoint (its name under `checkpoints/`) and the file in it that
 * holds the weights, the dataset, and for a derived model the parent's checkpoint.
 */
export interface BuyerFiles {
  checkpoint: string;
  weightsFile: string;
  dataset: URL;
  parentCheckpoint?: string;
}

/**
 * Asks the platform to verify the model passport and its whole chain, then reads the model
 * passport, the evaluation it cites, the event that produced the model, and the dataset passport
 * (found in the platform's tree by the dataset id the evaluation names).
 */
export async function readChain(client: AqClient, modelPassportUrl: string): Promise<ChainReading> {
  const platform = await client.verify("dpp", modelPassportUrl);
  const model = decodeJwtPayload(await client.fetchCredential(modelPassportUrl));
  const links = modelLinks(model);
  const evaluationJwt = await client.fetchCredential(links.evaluation.linkURL);
  const evaluation = decodeJwtPayload(evaluationJwt);
  const datasetId = evaluation.credentialSubject.conformityAssessment[0].evaluationRun.datasetId;
  const datasetLink = findPassportLink(platform, datasetId);
  if (!datasetLink)
    throw new Error(`no passport for the evaluated dataset ${datasetId} in the chain`);
  const datasetJwt = await client.fetchCredential(datasetLink.linkURL);
  const event = links.event
    ? decodeJwtPayload(await client.fetchCredential(links.event.linkURL))
    : undefined;

  const claims = readChainClaims(model, decodeJwtPayload(datasetJwt), evaluation, event);
  return {
    modelPassportUrl,
    platform,
    claims,
    credentialCheck: credentialDigestCheck([
      { link: datasetLink, jwt: datasetJwt },
      { link: links.evaluation, jwt: evaluationJwt },
    ]),
    inheritance: claims.derivation
      ? await readInheritance(client, platform, model, claims.derivation.parentId)
      : undefined,
  };
}

/** Reads the parent's passport and every DCC the derived passport cites, and classifies its claims. */
async function readInheritance(
  client: AqClient,
  platform: VerificationResult,
  // biome-ignore lint/suspicious/noExplicitAny: credentials are read loosely, like any JSON
  modelPassport: any,
  parentId: string,
): Promise<Inheritance> {
  const parentLink = findPassportLink(platform, parentId);
  if (!parentLink) throw new Error(`no passport for the parent model ${parentId} in the chain`);
  const parentPassport = decodeJwtPayload(await client.fetchCredential(parentLink.linkURL));
  const claims = classifyClaims(modelPassport, parentPassport);
  const assessed: Record<string, string[]> = {};
  for (const url of new Set(claims.flatMap((claim) => claim.evidence))) {
    assessed[url] = assessedProducts(decodeJwtPayload(await client.fetchCredential(url)));
  }
  return { modelPassport, parentPassport, claims, assessed };
}

/** The link that reached the passport for this product, anywhere in the platform's tree. */
function findPassportLink(result: VerificationResult, productId: string): Link | undefined {
  for (const child of result.upstreamResults ?? []) {
    if (child.credentialType === "dpp" && child.subject === productId && child.link) {
      return {
        linkURL: child.link.url,
        linkName: child.link.linkName,
        linkType: child.link.linkType ?? "",
        digestMultibase: child.link.digestMultibase,
      };
    }
    const found = findPassportLink(child, productId);
    if (found) return found;
  }
  return undefined;
}

/** The checks on the buyer's own files: hashes, a rerun of the evaluation, and of a derivation. */
export function fileChecks(reading: ChainReading, files: BuyerFiles): DemoCheck[] {
  const dir = new URL(`checkpoints/${files.checkpoint}/`, REPO_ROOT);
  const digests = {
    weights: hashFile(new URL(files.weightsFile, dir), "application/octet-stream"),
    config: hashFile(new URL("config.json", dir), "application/json"),
    dataset: hashFile(files.dataset, "text/plain"),
  };
  const measured = rerunEvaluation(files);
  return [
    ...artifactChecks(reading.claims, digests),
    evaluationRerunCheck(reading.claims, measured, harnessVersion()),
    ...derivationChecks(reading.claims, files),
  ];
}

/**
 * For a derived model: compare its weights with the parent's, and for a quantization, rerun it
 * from the parent's weights and compare hashes. Fine-tuning can't be rerun bit for bit, so the
 * comparison is all a buyer can check for it.
 */
export function derivationChecks(claims: ChainClaims, files: BuyerFiles): DemoCheck[] {
  const derivation = claims.derivation;
  if (!derivation) return [];
  if (!files.parentCheckpoint) {
    throw new Error("the parent's checkpoint is needed to check a derivation");
  }
  const parentDir = new URL(`checkpoints/${files.parentCheckpoint}/`, REPO_ROOT);
  const parentBytes = readFileSync(new URL("weights.bin", parentDir));
  const parent = floats(parentBytes);
  const shapes: number[][] = JSON.parse(
    readFileSync(new URL("config.json", parentDir), "utf8"),
  ).shapes;
  const checks: DemoCheck[] = [];

  if (derivation.processType.startsWith("Quantization")) {
    const rerun = isDeepStrictEqual(derivation.hyperparameters, { ...QUANTIZATION })
      ? sha256Hex(quantizeWeights(parent, shapes))
      : undefined;
    checks.push(
      ...quantizationRerunChecks(derivation, claims.modelId, sha256Hex(parentBytes), rerun),
    );
  }
  const comparison = weightComparisonCheck(
    derivation,
    weightChangeProfile(parent, derivedFloats(files, shapes), shapes),
  );
  if (comparison) checks.push(comparison);
  return checks;
}

/** The derived model's weights as float32 values, dequantized if they are int8. */
function derivedFloats(files: BuyerFiles, shapes: number[][]): Float32Array {
  const dir = new URL(`checkpoints/${files.checkpoint}/`, REPO_ROOT);
  const bytes = readFileSync(new URL(files.weightsFile, dir));
  return files.weightsFile === QUANTIZED_WEIGHTS ? dequantizeWeights(bytes, shapes) : floats(bytes);
}

function floats(bytes: Buffer): Float32Array {
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/** The checks that need only the credentials: they agree, and a derived model inherits fairly. */
export function readingChecks(reading: ChainReading): DemoCheck[] {
  const { inheritance, claims } = reading;
  return [
    reading.credentialCheck,
    ...claimChecks(claims),
    ...(inheritance && claims.derivation
      ? [inheritanceCheck(inheritance.claims, inheritance.assessed, claims.derivation.parentId)]
      : []),
  ];
}

/** Every check, in report order. */
export function allChecks(reading: ChainReading, files: BuyerFiles): DemoCheck[] {
  return [...readingChecks(reading), ...fileChecks(reading, files)];
}

/** The validation loss of the checkpoint on the dataset's validation split. */
export function rerunEvaluation(files: BuyerFiles): number {
  const gpt = isQuantized(files.checkpoint)
    ? loadQuantized(files.checkpoint)
    : loadCheckpoint(files.checkpoint);
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
 * The platform's result as a tree, one line per credential, with any check that didn't pass. A
 * credential linked from several places is shown in full once, and marked where it repeats.
 */
export function printPlatformTree(
  result: VerificationResult,
  indent = "",
  shown = new Set<string>(),
): void {
  const name = result.name ?? result.link?.linkName ?? result.credentialId;
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
    printPlatformTree(child, `${indent}  `, shown);
  }
}

/** A derived model's claims: inherited from the parent, overriding one of its claims, or new. */
export function printClaims(inheritance: Inheritance): void {
  for (const claim of inheritance.claims) {
    const parent = claim.parentClaim ? ` (parent's: ${claim.parentClaim})` : "";
    console.log(`  ${claim.origin.padEnd(9)} ${claim.name}${parent}`);
  }
}

export function printChecks(checks: DemoCheck[]): void {
  for (const check of checks) {
    console.log(`  ${check.passed ? "pass" : "FAIL"}  ${check.group.padEnd(11)} ${check.name}`);
    console.log(`        ${check.detail}`);
  }
}
