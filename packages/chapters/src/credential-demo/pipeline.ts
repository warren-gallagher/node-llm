import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { evaluateLoss, formatInput, formatResponse, type GptModel } from "@node-llm/core";
import { encode } from "gpt-tokenizer/encoding/r50k_base";
import { loadVerdict, makeLoaders } from "../ch05-training/shared";
import { loadInstructionData } from "../ch07-instruction/shared";
import { EVALUATION_BATCH_SIZE } from "./buyer";
import type {
  DatasetFacts,
  DatasetLicence,
  EvaluationFacts,
  ModelFacts,
  TrainingFacts,
} from "./claims";
import type { VerificationResult } from "./client";
import { REPO_ROOT } from "./env";
import { digestMultibase, hashFile, sha256Hex } from "./hashing";
import type { PartyKey } from "./parties";
import type { Manifest, RunRecord } from "./state";

/** Steps shared by the demo's pipeline scripts (`demo:run`, `demo:quantize`). */

const VERDICT_SOURCE_URL =
  "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";
const LOSS_METRIC = "cross-entropy loss (nats per token)";

export function authorityId(manifest: Manifest, key: PartyKey): string {
  const record = manifest.authorities[key];
  if (!record) throw new Error(`no authority for ${key}: run pnpm demo:setup`);
  return record.id;
}

/** The repository (as an https URL) and commit, warning if the working tree has changes. */
export function gitCode(): TrainingFacts["code"] {
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

export function printVerification(result: VerificationResult): void {
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

/** The Verdict as a dataset: its hash, size, token count and tokenizer. */
export function datasetFacts(): DatasetFacts {
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

/**
 * The Model Auditor's measurement: the validation loss of a loaded model with the shared harness,
 * on the last 10% of The Verdict.
 */
export function measureEvaluation(gpt: GptModel, code: TrainingFacts["code"]): EvaluationFacts {
  const { contextLength } = gpt.config;
  const { val } = makeLoaders(loadVerdict(), contextLength, EVALUATION_BATCH_SIZE);
  const validationLoss = evaluateLoss(gpt, val);
  console.log(`\nEvaluation: validation loss ${validationLoss}`);

  const corePackage = JSON.parse(
    readFileSync(new URL("packages/core/package.json", REPO_ROOT), "utf8"),
  );
  return {
    validationLoss,
    harness: { name: "@node-llm/core evaluateLoss", version: corePackage.version, ...code },
    split: `validation: the last 10% of the text's characters, in ${contextLength}-token windows`,
    metric: LOSS_METRIC,
  };
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

/** A run's model as its passport described it, rebuilt from its checkpoint. */
export function parentModelFacts(run: RunRecord): ModelFacts {
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

function parameterCount(shapes: number[][]): number {
  return shapes.reduce((sum, shape) => sum + shape.reduce((a, b) => a * b, 1), 0);
}

export const INSTRUCTION_DATA = new URL("data/instruction/instruction-data.json", REPO_ROOT);
const INSTRUCTION_SOURCE_URL =
  "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json";
const REPOSITORY_LICENCE: DatasetLicence = {
  id: "https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt",
  name: "Apache License 2.0, as modified in the LLMs-from-scratch repository's LICENSE.txt",
  rationale:
    "The repository's licence grants the right to use its contents, which this auditor " +
    "takes to include training models on the data.",
};

export /** Chapter 7's instruction data as a dataset: its file, source, licence and size in tokens. */
async function instructionFacts(verdict: DatasetFacts): Promise<DatasetFacts> {
  const entries = await loadInstructionData();
  const all = entries.map((entry) => formatInput(entry) + formatResponse(entry)).join("\n\n");
  return {
    title: "Instruction data (Build a Large Language Model (From Scratch), chapter 7)",
    description:
      "The 1,100 instruction-response examples the book's author prepared for chapter 7, as JSON.",
    file: hashFile(INSTRUCTION_DATA, "application/json"),
    sourceUrl: INSTRUCTION_SOURCE_URL,
    tokenCount: encode(all).length,
    tokenizer: verdict.tokenizer,
    licence: REPOSITORY_LICENCE,
  };
}
