import { classify, loadClassifier } from "./shared";

/**
 * Chapter 6, lesson 7: use the fine-tuned model.
 *
 * Loads the weights lesson 6 saved and classifies new messages. Each message is prepared
 * the same way as the training data: tokenized, then cut or padded to the training length.
 */
async function main(): Promise<void> {
  const saved = await loadClassifier("spam-classifier");
  if (!saved) {
    console.log("No fine-tuned model yet: run lesson 6 first (pnpm ch06:6).");
    return;
  }
  const { model, maxLength } = saved;

  const messages = [
    "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
    "Hey, just wanted to check if we're still on for dinner tonight? Let me know!",
    "URGENT! Your mobile number has been awarded a £500 prize. Call 09061701461 to claim.",
    "Can you pick up some milk on the way home?",
    "Congratulations, your package is waiting. Click the link to confirm delivery details.",
  ];
  for (const text of messages) {
    console.log(`${classify(model, text, maxLength).padEnd(8)}  ${JSON.stringify(text)}`);
  }

  model.dispose();
}

main();
