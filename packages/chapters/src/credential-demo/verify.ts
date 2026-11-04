import {
  allChecks,
  checkpointName,
  credentialNames,
  pickRun,
  printChecks,
  printPlatformTree,
  readChain,
  VERDICT_PATH,
} from "./buyer";
import { AqClient } from "./client";
import { loadDemoEnv } from "./env";
import { loadManifest } from "./state";

/**
 * Credential demo, the buyer's check (Model Deployment Inc.) of one run's model:
 *
 * 1. the platform verifies the model passport and everything it links, through the whole chain;
 * 2. the buyer reads the model passport, the dataset passport and the evaluation, and checks they
 *    agree with each other;
 * 3. the buyer hashes the model files and the dataset, and compares them with the passports;
 * 4. the buyer reruns the evaluation on those files, and compares the loss with the claimed one.
 *
 * Usage: pnpm demo:verify [runId]   (default: the latest run)
 */

async function main(): Promise<void> {
  const env = loadDemoEnv();
  const client = new AqClient(env);
  const run = pickRun(loadManifest(env.apiUrl, env.tenantId), process.argv[2]);

  console.log(`Run ${run.runId}`);
  console.log(`Model passport: ${run.passport.credentialId}`);
  const reading = await readChain(client, run.passport.credentialId);

  console.log("\nPlatform verification, through the whole chain:");
  printPlatformTree(reading.platform, await credentialNames(client, reading.platform), "  ");

  console.log("\nThe buyer's checks:");
  const checks = allChecks(reading, { checkpoint: checkpointName(run), dataset: VERDICT_PATH });
  printChecks(checks);

  const accepted = reading.platform.verified && checks.every((check) => check.passed);
  console.log(
    `\n${accepted ? "ACCEPTED" : "NOT ACCEPTED"}: the platform verification and every check above.`,
  );
  console.log(
    "Not checked: whether each issuer should be trusted for its role (see 'Shortcoming: trust' in docs/credential-demo/credential-demo.md).",
  );
  if (!accepted) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
