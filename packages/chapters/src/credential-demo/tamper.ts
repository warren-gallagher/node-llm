import { cpSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import {
  allChecks,
  type BuyerFiles,
  type BuyerModel,
  type ChainReading,
  pickModel,
  readChain,
  VERDICT_PATH,
} from "./buyer";
import {
  assessedProducts,
  type CheckGroup,
  changeOneWeight,
  changeOneWord,
  classifyClaims,
  type DemoCheck,
  decodeJwtPayload,
} from "./checks";
import { AqClient } from "./client";
import { loadDemoEnv, REPO_ROOT } from "./env";
import { digestMultibase, niUri, sha256Hex } from "./hashing";
import { QUANTIZATION } from "./quantize";
import { loadManifest } from "./state";

/**
 * Credential demo, failure cases, and which of the buyer's checks catches each.
 *
 * For a trained model, the buyer's files are altered; the credentials are untouched, so the
 * platform verifies them every time: it can't see the files.
 *
 * For a derived model, the cases are lies in the credentials themselves: a derivation that hides
 * an edit, or claims the wrong method, or a passport that inherits a claim it shouldn't. Issuing
 * false credentials would publish them, so these cases are simulated: what the buyer read from
 * the credentials is changed in memory, and nothing is issued. The platform column shows the
 * real credentials' verification.
 *
 * Usage: pnpm demo:tamper [runId | derivationId]   (default: the latest run)
 */

const TAMPER_DIR = "verdict-demo/tamper";
const GROUPS: CheckGroup[] = ["credentials", "claims", "artifact", "evaluation", "derivation"];

interface TamperCase {
  name: string;
  /** The check groups that should fail, and no others. */
  expectedFailures: CheckGroup[];
  note: string;
  /** The files the buyer checks, and what it read from the credentials if that changes. */
  prepare(
    model: BuyerModel,
    reading: ChainReading,
    client: AqClient,
  ): Promise<{ files: BuyerFiles; reading?: ChainReading }>;
}

const UNTAMPERED: TamperCase = {
  name: "Untampered",
  expectedFailures: [],
  note: "the files and credentials as issued",
  async prepare(model) {
    return { files: model.files };
  },
};

const TRAINED_MODEL_CASES: TamperCase[] = [
  UNTAMPERED,
  {
    name: "One weight changed",
    expectedFailures: ["artifact"],
    note: "the first weight is 0.001 larger; the loss doesn't measurably change, so only the hash catches it",
    async prepare(model) {
      const checkpoint = copyCheckpoint(model, "one-weight");
      const weights = new URL(model.files.weightsFile, checkpointUrl(checkpoint));
      writeFileSync(weights, changeOneWeight(readFileSync(weights)));
      return { files: { ...model.files, checkpoint } };
    },
  },
  {
    name: "A different dataset file",
    expectedFailures: ["artifact"],
    note: "one name changed in the training part of the text, so the validation loss is the same",
    async prepare(model) {
      const dir = checkpointUrl(`${TAMPER_DIR}/dataset`);
      mkdirSync(dir, { recursive: true });
      const dataset = new URL("the-verdict.txt", dir);
      writeFileSync(dataset, changeOneWord(readFileSync(VERDICT_PATH, "utf8")));
      return { files: { ...model.files, dataset } };
    },
  },
];

const QUANTIZATION_CASES: TamperCase[] = [
  UNTAMPERED,
  {
    name: "A hidden edit passed off as a quantization (simulated)",
    expectedFailures: ["derivation"],
    note:
      "one int8 value moved 5 steps after quantizing, and the credentials describe the edited " +
      "file, so the hashes match; rerunning the quantization doesn't",
    async prepare(model, reading) {
      const checkpoint = copyCheckpoint(model, "hidden-edit");
      const weights = new URL(model.files.weightsFile, checkpointUrl(checkpoint));
      const bytes = readFileSync(weights);
      // Byte 4 is the first value of the first tensor, after its float32 scale.
      bytes.writeInt8(Math.min(127, bytes.readInt8(4) + 5), 4);
      writeFileSync(weights, bytes);
      return {
        files: { ...model.files, checkpoint },
        reading: describing(reading, model.files.weightsFile, sha256Hex(bytes)),
      };
    },
  },
];

const FINE_TUNING_CASES: TamperCase[] = [
  UNTAMPERED,
  {
    name: "A fine-tune passed off as a quantization (simulated)",
    expectedFailures: ["derivation"],
    note: "the derivation event says it quantized the parent, but the weights were fine-tuned",
    async prepare(model, reading) {
      const derivation = reading.claims.derivation;
      if (!derivation) throw new Error("not a derived model");
      const claims = {
        ...reading.claims,
        derivation: {
          ...derivation,
          processType: "Quantization (int8)",
          hyperparameters: { ...QUANTIZATION },
        },
      };
      return { files: model.files, reading: { ...reading, claims } };
    },
  },
  {
    name: "A passport that inherits the parent's evaluation (simulated)",
    expectedFailures: ["claims"],
    note: "the loss claim cites the parent's evaluation instead of the fine-tuned model's own",
    async prepare(model, reading, client) {
      return { files: model.files, reading: await inheritingParentEvaluation(reading, client) };
    },
  },
];

const CASES: Record<BuyerModel["kind"], TamperCase[]> = {
  run: TRAINED_MODEL_CASES,
  quantization: QUANTIZATION_CASES,
  "fine-tuning": FINE_TUNING_CASES,
};

async function main(): Promise<void> {
  const env = loadDemoEnv();
  const client = new AqClient(env);
  const model = pickModel(loadManifest(env.apiUrl, env.tenantId), process.argv[2]);

  console.log(`Model ${model.id} (${model.kind})`);
  const reading = await readChain(client, model.passport.credentialId);
  const platform = reading.platform.verified ? "VERIFIED" : "NOT VERIFIED";
  const groups = GROUPS.filter((group) => group !== "derivation" || model.kind !== "run");

  const rows: string[][] = [];
  let allAsExpected = true;
  for (const tamperCase of CASES[model.kind]) {
    console.log(`\n${tamperCase.name}: ${tamperCase.note}`);
    const prepared = await tamperCase.prepare(model, reading, client);
    const checks = allChecks(prepared.reading ?? reading, prepared.files);
    const rerun = checks.find((check) => check.group === "evaluation");
    if (rerun) console.log(`  evaluation rerun: ${rerun.detail}`);
    for (const check of checks.filter((c) => !c.passed)) {
      console.log(`  FAIL ${check.group}: ${check.name}`);
      console.log(`       ${check.detail}`);
    }
    const failed = failedGroups(checks);
    const asExpected =
      failed.length === tamperCase.expectedFailures.length &&
      failed.every((group) => tamperCase.expectedFailures.includes(group));
    allAsExpected &&= asExpected;
    rows.push([
      tamperCase.name,
      platform,
      ...groups.map((group) => (failed.includes(group) ? "FAIL" : "pass")),
      failed.length > 0 ? failed.join(", ") : "nothing (accepted)",
      asExpected ? "yes" : "NO",
    ]);
  }

  console.log("");
  printTable(["Case", "Platform", ...groups.map(capitalize), "Caught by", "As expected"], rows);
  if (!allAsExpected) process.exitCode = 1;
}

/** A copy of the model's checkpoint under the tamper directory, by name. */
function copyCheckpoint(model: BuyerModel, name: string): string {
  const checkpoint = `${TAMPER_DIR}/${name}`;
  cpSync(checkpointUrl(model.files.checkpoint), checkpointUrl(checkpoint), { recursive: true });
  return checkpoint;
}

/** What the buyer would read if the credentials described a weights file with this hash. */
function describing(reading: ChainReading, weightsFile: string, sha256: string): ChainReading {
  const id = niUri(sha256);
  const { claims } = reading;
  return {
    ...reading,
    claims: {
      ...claims,
      modelId: id,
      modelFiles: claims.modelFiles.map((file) =>
        file.name === weightsFile ? { ...file, digestMultibase: digestMultibase(sha256) } : file,
      ),
      evaluation: { ...claims.evaluation, assessedProductId: id },
      derivation: claims.derivation && { ...claims.derivation, derivedId: id },
    },
  };
}

/** What the buyer would read if the passport's loss claim cited the parent's evaluation. */
async function inheritingParentEvaluation(
  reading: ChainReading,
  client: AqClient,
): Promise<ChainReading> {
  const { inheritance } = reading;
  if (!inheritance) throw new Error("not a derived model");
  const passport = structuredClone(inheritance.modelPassport);
  const parentLoss =
    inheritance.parentPassport.credentialSubject.performanceClaim.find(isLossClaim);
  const loss = passport.credentialSubject.performanceClaim.find(isLossClaim);
  loss.evidence = parentLoss.evidence;

  const assessed = { ...inheritance.assessed };
  for (const link of parentLoss.evidence) {
    assessed[link.linkURL] ??= assessedProducts(
      decodeJwtPayload(await client.fetchCredential(link.linkURL)),
    );
  }
  return {
    ...reading,
    inheritance: {
      ...inheritance,
      modelPassport: passport,
      claims: classifyClaims(passport, inheritance.parentPassport),
      assessed,
    },
  };
}

/** The claim that carries a measured value: the loss claim. */
function isLossClaim(claim: { claimedPerformance?: { measure?: unknown }[] }): boolean {
  return claim.claimedPerformance?.[0]?.measure !== undefined;
}

function checkpointUrl(name: string): URL {
  return new URL(`checkpoints/${name}/`, REPO_ROOT);
}

function failedGroups(checks: DemoCheck[]): CheckGroup[] {
  return GROUPS.filter((group) => checks.some((check) => check.group === group && !check.passed));
}

function capitalize(word: string): string {
  return word[0].toUpperCase() + word.slice(1);
}

function printTable(header: string[], rows: string[][]): void {
  const widths = header.map((cell, i) =>
    Math.max(cell.length, ...rows.map((row) => row[i].length)),
  );
  for (const row of [header, ...rows]) {
    console.log(
      row
        .map((cell, i) => cell.padEnd(widths[i]))
        .join("  ")
        .trimEnd(),
    );
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
