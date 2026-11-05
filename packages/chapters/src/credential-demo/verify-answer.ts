import { loadFineTuned } from "../ch07-instruction/shared";
import { answerChecks, answerFiles, pickAnswer, readAnswer } from "./answer-check";
import { printChecks, printPlatformTree } from "./buyer";
import { AqClient } from "./client";
import { loadDemoEnv, REPO_ROOT } from "./env";
import { hashFile } from "./hashing";
import { loadEmbedder, readCollection } from "./rag";
import { loadManifest } from "./state";

/**
 * Credential demo, phase 3: a buyer checks one answer.
 *
 * 1. The platform verifies the answer's DTE and everything it links: the answering model and its
 *    lineage back to GPT-2, and the index back to the collection and the embedding model.
 * 2. The buyer checks that the question and answer it holds are the ones recorded, that every
 *    passage the answer rests on is in Book Publisher's collection passport, and that its copy of
 *    the index is the one recorded, holds only listed passages, and rebuilds the same.
 * 3. The buyer retrieves again and generates again, and gets the recorded passage and answer.
 *
 * That shows where the answer came from. It doesn't show the answer is right: a model can
 * misread the right passage.
 *
 * Usage: pnpm demo:verify-answer [answerId]   (default: the latest answer)
 */

async function main(): Promise<void> {
  const env = loadDemoEnv();
  const client = new AqClient(env);
  const manifest = loadManifest(env.apiUrl, env.tenantId);
  const entry = pickAnswer(manifest, process.argv[2]);

  console.log(`Answer ${entry.answerId}`);
  console.log(`  question: ${entry.question}`);
  console.log(`  answer:   ${entry.answer}`);
  console.log(`  record:   ${entry.event.credentialId}`);
  const reading = await readAnswer(client, entry.event.credentialId);

  console.log("\nPlatform verification, through the whole chain:");
  printPlatformTree(reading.platform, "  ");

  const embedder = await loadEmbedder();
  const generator = loadFineTuned("124m");
  if (!generator) throw new Error("no instruction-tuned GPT-2 124M: run pnpm ch07:5");
  const generatorSha256 = hashFile(
    new URL("checkpoints/gpt2-124m-instruct/weights.bin", REPO_ROOT),
    "application/octet-stream",
  ).sha256;
  const checks = await answerChecks(
    reading,
    answerFiles(entry, manifest, readCollection()),
    { embedder, generator, generatorSha256 },
    manifest.authorities.bookPublisher?.did,
  );
  generator.dispose();
  await embedder.dispose();

  console.log("\nThe buyer's checks:");
  printChecks(checks);
  const accepted = reading.platform.verified && checks.every((check) => check.passed);
  console.log(
    `\n${accepted ? "ACCEPTED" : "NOT ACCEPTED"}: the platform verification and every check above.`,
  );
  console.log("This shows where the answer came from, not that it is right.");
  if (!accepted) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
