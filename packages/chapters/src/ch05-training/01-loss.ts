import { crossEntropyLoss, evaluateLoss, GptModel } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";
import { decode, encode } from "gpt-tokenizer/encoding/r50k_base";
import { BOOK_CONFIG, loadVerdict, makeLoaders } from "./shared";

/**
 * Chapter 5, Lesson 1: measuring how wrong the model is with cross-entropy loss.
 *
 * 1. Two short inputs and their targets (the same text shifted one token). Compute the
 *    loss by hand, step by step, then check it against `crossEntropyLoss`.
 * 2. The untrained model's loss over the whole training text and validation text.
 */
function main(): void {
  const model = new GptModel(BOOK_CONFIG);

  // 1. Inputs and targets: each target is the next token after the matching input token.
  const inputs = tf.tensor2d(
    [encode("every effort moves"), encode("I really like")],
    undefined,
    "int32",
  );
  const targets = tf.tensor2d(
    [encode(" effort moves you"), encode(" really like chocolate")],
    undefined,
    "int32",
  );
  console.log("inputs: ", inputs.arraySync());
  console.log("targets:", targets.arraySync());

  const probs = tf.softmax(model.forward(inputs), -1); // [2, 3, 50257]

  // What an untrained model would predict: the highest-probability token at each position.
  const predicted = tf.argMax(probs, -1).arraySync() as number[][];
  console.log(
    "\nuntrained predictions:",
    predicted.map((row) => JSON.stringify(decode(row))),
  );

  // The probability the model gave each correct target token.
  const targetProbs = targetProbabilities(probs, targets);
  console.log(
    "\nprobability of each correct token:",
    targetProbs.map((p) => p.toExponential(2)),
  );

  const logProbs = targetProbs.map(Math.log);
  const meanLogProb = logProbs.reduce((sum, v) => sum + v, 0) / logProbs.length;
  console.log(
    "their logs:",
    logProbs.map((v) => v.toFixed(3)),
  );
  console.log("mean log probability:", meanLogProb.toFixed(4));
  console.log("cross-entropy (negated mean):", (-meanLogProb).toFixed(4));

  const loss = crossEntropyLoss(model, inputs, targets).dataSync()[0] as number;
  console.log("crossEntropyLoss():", loss.toFixed(4));
  console.log("perplexity, exp(loss):", Math.round(Math.exp(loss)).toLocaleString());
  console.log("uniform-guess loss, ln(50257):", Math.log(BOOK_CONFIG.vocabSize).toFixed(4));

  // 2. The same measure over all the training and validation text.
  const { train, val, trainTokens, valTokens } = makeLoaders(
    loadVerdict(),
    BOOK_CONFIG.contextLength,
  );
  console.log(`\ntraining tokens: ${trainTokens}, validation tokens: ${valTokens}`);
  console.log("training loss:  ", evaluateLoss(model, train).toFixed(4));
  console.log("validation loss:", evaluateLoss(model, val).toFixed(4));

  tf.dispose([inputs, targets, probs]);
  model.dispose();
}

/** probs: [batch, seqLen, vocabSize], targets: [batch, seqLen] -> one probability per target. */
function targetProbabilities(probs: tf.Tensor3D, targets: tf.Tensor2D): number[] {
  const p = probs.arraySync();
  const t = targets.arraySync();
  return t.flatMap((row, b) => row.map((id, pos) => p[b]?.[pos]?.[id] as number));
}

main();
