import { allChecks, pickModel, printChecks, printPlatformTree, readChain } from "./buyer";
import { AqClient } from "./client";
import { loadDemoEnv } from "./env";
import { loadManifest } from "./state";

/**
 * Credential demo, the buyer's check (Model Deployment Inc.) of one model, trained by a run or
 * derived from a run's model:
 *
 * 1. the platform verifies the model passport and everything it links, through the whole chain;
 * 2. the buyer reads the model passport, the dataset passport and the evaluation, and checks they
 *    agree with each other;
 * 3. the buyer hashes the model files and the dataset, and compares them with the passports;
 * 4. the buyer reruns the evaluation on those files, and compares the loss with the claimed one;
 * 5. for a quantized model, the buyer reruns the quantization from the parent's weights.
 *
 * Usage: pnpm demo:verify [runId | derivationId]   (default: the latest run)
 */

async function main(): Promise<void> {
  const env = loadDemoEnv();
  const client = new AqClient(env);
  const model = pickModel(loadManifest(env.apiUrl, env.tenantId), process.argv[2]);

  console.log(`Model ${model.id}`);
  console.log(`Model passport: ${model.passport.credentialId}`);
  const reading = await readChain(client, model.passport.credentialId);

  console.log("\nPlatform verification, through the whole chain:");
  printPlatformTree(reading.platform, "  ");

  console.log("\nThe buyer's checks:");
  const checks = allChecks(reading, model.files);
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
