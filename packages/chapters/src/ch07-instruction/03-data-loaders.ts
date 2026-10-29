import { decode } from "gpt-tokenizer/encoding/r50k_base";
import { makeInstructionLoaders } from "./shared";

/**
 * Chapter 7, lesson 3: data loaders for the instruction dataset.
 *
 * Batches of 8. The training loader shuffles and drops the last, incomplete batch; the
 * validation and test loaders keep their order. Because each batch is padded only to its own
 * longest example, batch lengths vary, which saves computation compared with padding every
 * example to the longest in the dataset.
 */
async function main(): Promise<void> {
  const { trainLoader, validationLoader, testLoader } = await makeInstructionLoaders();
  console.log(
    `batches: training ${trainLoader.length}, validation ${validationLoader.length}, test ${testLoader.length}`,
  );

  const lengths = [...trainLoader].map((b) => b.inputs[0]?.length ?? 0);
  console.log(`\ntraining batch shapes (shuffled, so the order changes each run):`);
  console.log(
    lengths
      .slice(0, 8)
      .map((n) => `[8, ${n}]`)
      .join(" "),
    "...",
  );
  console.log(`shortest [8, ${Math.min(...lengths)}], longest [8, ${Math.max(...lengths)}]`);

  // The validation loader isn't shuffled, so this is the same example on every run.
  const [first] = validationLoader;
  if (!first) return;
  const lengthsInBatch = first.targets.map((row) => row.filter((id) => id !== -100).length);
  const row = lengthsInBatch.indexOf(Math.min(...lengthsInBatch)); // the most padded example
  const input = first.inputs[row] ?? [];
  const target = first.targets[row] ?? [];

  console.log(`\nvalidation batch 1, example ${row + 1} (shape [8, ${input.length}]):`);
  console.log("inputs: ", input.join(" "));
  console.log("targets:", target.join(" "));
  console.log("\nas text:");
  console.log(decode(input));
}

main();
