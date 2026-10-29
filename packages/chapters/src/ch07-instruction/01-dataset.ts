import { formatInput, formatResponse } from "@node-llm/core";
import { loadInstructionData, loadInstructionSplits } from "./shared";

/**
 * Chapter 7, lesson 1: the instruction dataset.
 *
 * Each example is an instruction, an optional input, and the desired output. To train on
 * one, it's written out as a single text in a fixed template (here the Alpaca template), so
 * the model learns to continue a prompt in that template with a response.
 */
async function main(): Promise<void> {
  const data = await loadInstructionData();
  console.log(`number of entries: ${data.length}\n`);

  for (const index of [50, 999]) {
    const entry = data[index];
    if (!entry) continue;
    console.log(`entry ${index}:`, entry);
    console.log("\nas one training text:");
    console.log("-".repeat(60));
    console.log(formatInput(entry) + formatResponse(entry));
    console.log(`${"-".repeat(60)}\n`);
  }

  const { train, validation, test } = await loadInstructionSplits();
  console.log(`training set:   ${train.length}`);
  console.log(`validation set: ${validation.length}`);
  console.log(`test set:       ${test.length}`);
}

main();
