import type { InstructionEntry } from "@node-llm/core";
import { loadFineTuned, loadInstructionSplits } from "../ch07-instruction/shared";
import { answer, readIndex, textDigest } from "./answers";
import { answerDte, type DemoContext } from "./claims";
import { AqClient } from "./client";
import { loadDemoEnv, REPO_ROOT } from "./env";
import { hashFile } from "./hashing";
import { authorityId, printVerification } from "./pipeline";
import { collectionPassages, loadEmbedder } from "./rag";
import {
  type AnswerRecordEntry,
  loadManifest,
  partyIdentities,
  saveManifest,
  toIssued,
} from "./state";

/**
 * Credential demo, phase 3, step 2: answers. Model Deployment Inc. answers chapter 7's six
 * retrieval questions: it retrieves the closest passage from its index, has the instruction-tuned
 * GPT-2 answer from it, and records each answer in its own DTE (the question, the passage and the
 * answer by the hashes of their texts, with the model and the index). One DTE per answer suits a
 * demo; a real system answering at volume would record answers in bulk.
 *
 * Usage: pnpm demo:ask [--dry-run]
 */

/** The test-set questions chapter 7's retrieval lesson asks (its numbering, from 0). */
const QUESTIONS = [2, 13, 3, 1, 39, 24];
const GENERATOR_NAME = "GPT-2 124M instruction-tuned (chapter 7)";
const INDEX_NAME = "Reference passages index (all-MiniLM-L6-v2)";

async function main(): Promise<void> {
  const env = loadDemoEnv();
  const client = new AqClient(env);
  const manifest = loadManifest(env.apiUrl, env.tenantId);
  const dryRun = process.argv.includes("--dry-run");
  const rag = manifest.rag;
  if (!rag?.generator || !rag.index)
    throw new Error("no retrieval system yet: run pnpm demo:rag-setup");
  const ctx: DemoContext = {
    apiUrl: env.apiUrl,
    parties: partyIdentities(manifest),
    date: new Date().toISOString().slice(0, 10),
  };
  if (dryRun) console.log("Dry run: the platform validates and signs, but publishes nothing\n");

  // The model and the index the credentials describe, checked by hash before use.
  const weights = hashFile(
    new URL("checkpoints/gpt2-124m-instruct/weights.bin", REPO_ROOT),
    "application/octet-stream",
  );
  if (weights.sha256 !== rag.generator.sha256)
    throw new Error("the answering model changed since its passport");
  const indexPath = new URL(rag.index.file, REPO_ROOT);
  const indexFile = hashFile(indexPath, "application/json");
  if (indexFile.sha256 !== rag.index.sha256)
    throw new Error("the index changed since its passport");
  const generator = loadFineTuned("124m");
  if (!generator) throw new Error("no instruction-tuned GPT-2 124M: run pnpm ch07:5");
  const embedder = await loadEmbedder();
  const entries = readIndex(indexPath);
  const passages = collectionPassages();
  const { test } = await loadInstructionSplits();

  const answers: AnswerRecordEntry[] = [];
  for (const [n, i] of QUESTIONS.entries()) {
    const entry = test[i] as InstructionEntry;
    const facts = await answer(embedder, generator, entries, passages, entry);
    const [passage] = facts.passages;
    console.log(`\n${facts.question}`);
    console.log(`  retrieved ${passage?.id} (similarity ${passage?.score.toFixed(3)})`);
    console.log(`  answer: ${facts.answer}`);

    const eventDate = new Date().toISOString();
    const answerId = `${eventDate.slice(0, 19).replaceAll(":", "-")}Z-a${n + 1}`;
    const event = toIssued(
      await client.sign(
        "dte",
        authorityId(manifest, "modelDeployer"),
        answerDte(
          ctx,
          {
            answerId,
            eventDate,
            question: { ...textDigest("question", facts.question), text: facts.question },
            answer: { ...textDigest("answer", facts.answer), text: facts.answer },
            passages: facts.passages.map((p) => ({
              ...textDigest(p.id, passages.find((q) => q.id === p.id)?.text ?? ""),
              score: p.score,
            })),
          },
          { name: GENERATOR_NAME, weights },
          { name: INDEX_NAME, file: indexFile },
          { generatorPassport: rag.generator.passport, indexPassport: rag.index.passport },
        ),
        { store: !dryRun },
      ),
    );
    console.log(`  answer DTE (Model Deployment Inc.): ${event.credentialId}`);
    answers.push({ answerId, ...facts, event });
  }
  generator.dispose();
  await embedder.dispose();

  if (dryRun) {
    console.log("\nDry run complete: every answer record passed the platform's validation.");
    return;
  }
  rag.answers = [...(rag.answers ?? []), ...answers];
  saveManifest(manifest);
  console.log("\nPlatform verification of the last answer, through the whole chain:");
  const last = answers.at(-1);
  if (last) printVerification(await client.verify("dte", last.event.credentialId));
  console.log(`\nCheck one as a buyer: pnpm demo:verify-answer ${last?.answerId}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
