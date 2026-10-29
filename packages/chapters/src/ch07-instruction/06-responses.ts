import { writeFileSync } from "node:fs";
import { formatInput } from "@node-llm/core";
import { chosenSize, DATA_DIR, loadFineTuned, loadInstructionSplits, respond } from "./shared";

/**
 * Chapter 7, lesson 6: generate responses to the test set and save them.
 *
 * The fine-tuned model answers each of the 110 test instructions it never saw in training.
 * Lesson 7 scores the answers, so they're saved to
 * `data/instruction/test-responses-<size>.json`, each entry with a `model_response` field
 * added, as in the book.
 */
async function main(): Promise<void> {
  const size = chosenSize();
  const model = loadFineTuned(size);
  if (!model) {
    const flag = size === "355m" ? " --355m" : "";
    console.log(`no fine-tuned GPT-2 ${size} yet: run lesson 5 first (pnpm ch07:5${flag})`);
    return;
  }
  const { test } = await loadInstructionSplits();

  for (const entry of test.slice(0, 3)) {
    console.log(formatInput(entry));
    console.log(`\ncorrect response:\n>> ${entry.output}`);
    console.log(`\nmodel response:\n>> ${respond(model, entry)}`);
    console.log("-".repeat(40));
  }

  console.log(`\ngenerating responses to all ${test.length} test instructions...`);
  const start = Date.now();
  const withResponses = test.map((entry, i) => {
    const model_response = respond(model, entry);
    if ((i + 1) % 10 === 0) {
      const seconds = (Date.now() - start) / 1000;
      console.log(`  ${i + 1}/${test.length} (${seconds.toFixed(0)} s)`);
    }
    return { ...entry, model_response };
  });

  const file = new URL(`test-responses-${size}.json`, DATA_DIR);
  writeFileSync(file, JSON.stringify(withResponses, null, 4));
  console.log(`saved to ${file.pathname}`);
  model.dispose();
}

main();
