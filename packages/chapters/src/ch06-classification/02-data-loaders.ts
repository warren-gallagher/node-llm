import { decode, encode } from "gpt-tokenizer/encoding/r50k_base";
import { makeSpamLoaders } from "./shared";

/**
 * Chapter 6, lesson 2: padding and batching.
 *
 * Messages have different lengths, but a batch tensor needs equal-length rows. Every message
 * is padded with `<|endoftext|>` (id 50256) to the length of the longest training message.
 * Validation and test messages are padded, or cut, to that same length.
 */
async function main(): Promise<void> {
  console.log(`padding token: 50256 = ${JSON.stringify(decode([50256]))}`);

  const { train, validation, test, maxLength } = await makeSpamLoaders();
  console.log(`longest training message: ${maxLength} tokens`);

  const example = "Ok lar... Joking wif u oni...";
  const ids = encode(example);
  console.log(`\n${JSON.stringify(example)}: ${ids.length} tokens`);
  console.log(`  padded: [${ids.join(", ")}, 50256 x ${maxLength - ids.length}]`);

  const [batch] = train;
  if (batch) {
    console.log(`\ninput batch: [${batch.inputs.length}, ${batch.inputs[0]?.length}]`);
    console.log(`label batch: [${batch.labels.length}]  ${JSON.stringify(batch.labels)}`);
  }

  console.log(`\n${train.length} training batches (the last partial batch is dropped)`);
  console.log(`${validation.length} validation batches`);
  console.log(`${test.length} test batches`);
}

main();
