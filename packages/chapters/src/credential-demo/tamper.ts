import { cpSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import {
  allChecks,
  type BuyerFiles,
  checkpointName,
  pickRun,
  readChain,
  VERDICT_PATH,
} from "./buyer";
import { type CheckGroup, changeOneWeight, changeOneWord, type DemoCheck } from "./checks";
import { AqClient } from "./client";
import { loadDemoEnv, REPO_ROOT } from "./env";
import { loadManifest, type RunRecord } from "./state";

/**
 * Credential demo, failure cases. The buyer's check runs on altered copies of the latest run's
 * files, and the report shows which check catches each change. The credentials are untouched, so
 * the platform verifies them every time: it can't see the files.
 *
 * Usage: pnpm demo:tamper [runId]   (default: the latest run)
 */

const TAMPER_DIR = "verdict-demo/tamper";
const GROUPS: CheckGroup[] = ["credentials", "claims", "artifact", "evaluation"];

interface TamperCase {
  name: string;
  /** The check groups that should fail, and no others. */
  expectedFailures: CheckGroup[];
  note: string;
  prepare(run: RunRecord): BuyerFiles;
}

const CASES: TamperCase[] = [
  {
    name: "Untampered",
    expectedFailures: [],
    note: "the files the credentials describe",
    prepare(run) {
      return { checkpoint: checkpointName(run), weightsFile: "weights.bin", dataset: VERDICT_PATH };
    },
  },
  {
    name: "One weight changed",
    expectedFailures: ["artifact"],
    note: "the first weight is 0.001 larger; the loss doesn't measurably change, so only the hash catches it",
    prepare(run) {
      const checkpoint = `${TAMPER_DIR}/one-weight`;
      const dir = checkpointUrl(checkpoint);
      cpSync(checkpointUrl(checkpointName(run)), dir, { recursive: true });
      const weights = new URL("weights.bin", dir);
      writeFileSync(weights, changeOneWeight(readFileSync(weights)));
      return { checkpoint, weightsFile: "weights.bin", dataset: VERDICT_PATH };
    },
  },
  {
    name: "A different dataset file",
    expectedFailures: ["artifact"],
    note: "one name changed in the training part of the text, so the validation loss is the same",
    prepare(run) {
      const dir = checkpointUrl(`${TAMPER_DIR}/dataset`);
      mkdirSync(dir, { recursive: true });
      const dataset = new URL("the-verdict.txt", dir);
      writeFileSync(dataset, changeOneWord(readFileSync(VERDICT_PATH, "utf8")));
      return { checkpoint: checkpointName(run), weightsFile: "weights.bin", dataset };
    },
  },
];

async function main(): Promise<void> {
  const env = loadDemoEnv();
  const client = new AqClient(env);
  const run = pickRun(loadManifest(env.apiUrl, env.tenantId), process.argv[2]);

  console.log(`Run ${run.runId}`);
  const reading = await readChain(client, run.passport.credentialId);
  const platform = reading.platform.verified ? "VERIFIED" : "NOT VERIFIED";

  const rows: string[][] = [];
  let allAsExpected = true;
  for (const tamperCase of CASES) {
    console.log(`\n${tamperCase.name}: ${tamperCase.note}`);
    const checks = allChecks(reading, tamperCase.prepare(run));
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
      ...GROUPS.map((group) => (failed.includes(group) ? "FAIL" : "pass")),
      failed.length > 0 ? failed.join(", ") : "nothing (accepted)",
      asExpected ? "yes" : "NO",
    ]);
  }

  console.log("");
  printTable(["Case", "Platform", ...GROUPS.map(capitalize), "Caught by", "As expected"], rows);
  if (!allAsExpected) process.exitCode = 1;
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
