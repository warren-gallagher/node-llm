import * as tf from "@tensorflow/tfjs-node";
import { ensureGpt2Weights, loadGpt2 } from "../ch05-training/gpt2";
import { idsToText, textToIds } from "../ch05-training/shared";

/**
 * Chapter 6, lesson 3: start from pretrained GPT-2.
 *
 * Fine-tuning starts from a model that already knows language. First check that the weights
 * loaded correctly by generating text, then see whether the pretrained model can classify
 * spam just by being asked. It can't: it was trained to continue text, not to follow
 * instructions, so it continues the prompt instead of answering it.
 */
async function main(): Promise<void> {
  const model = loadGpt2(await ensureGpt2Weights(), { dropout: 0 });

  const prompts: [string, number][] = [
    ["Every effort moves you", 15],
    [
      "Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you have" +
        " been specially selected to receive $1000 cash or a $2000 award.'",
      23,
    ],
  ];
  for (const [text, newTokens] of prompts) {
    const prompt = textToIds(text);
    const out = model.generate(prompt, newTokens);
    console.log(`${JSON.stringify(idsToText(out))}\n`);
    tf.dispose([prompt, out]);
  }

  model.dispose();
}

main();
