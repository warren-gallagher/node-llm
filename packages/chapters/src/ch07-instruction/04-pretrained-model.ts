import { formatInput } from "@node-llm/core";
import { chosenSize, loadInstructionSplits, loadPretrained, respond } from "./shared";

/**
 * Chapter 7, lesson 4: load a pretrained GPT-2 and try it before fine-tuning.
 *
 * The book uses GPT-2 medium (355M parameters), because the 124M model is too small to
 * give reasonable answers after instruction fine-tuning. These lessons default to 124M so
 * they run quickly on a CPU; add `--355m` to use the book's size (a 1.5 GB download).
 *
 * The pretrained model has only learned to continue text, so given a prompt in the
 * instruction template it tends to repeat or continue the template, not answer.
 */
async function main(): Promise<void> {
  const size = chosenSize();
  const model = await loadPretrained(size);
  const { embDim, numLayers, numHeads } = model.config;
  console.log(
    `GPT-2 ${size.toUpperCase()}: ${model.numParameters().toLocaleString()} parameters, ` +
      `${numLayers} blocks, ${numHeads} heads, embeddings of ${embDim}`,
  );

  const { validation } = await loadInstructionSplits();
  const entry = validation[0];
  if (!entry) return;
  console.log(`\n${formatInput(entry)}`);
  console.log(`\nresponse before fine-tuning (up to 35 tokens):\n>> ${respond(model, entry, 35)}`);
  model.dispose();
}

main();
