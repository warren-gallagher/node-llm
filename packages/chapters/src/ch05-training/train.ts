import { DataLoader, GPT_CONFIG_124M, GptModel, LlmDataset, trainModel } from "@node-llm/core";
import "@tensorflow/tfjs-node";
import { encode as gptEncode } from "gpt-tokenizer";

/**
 * Chapter 5: end-to-end training loop on a tiny corpus.
 *
 * The bare `import "@tensorflow/tfjs-node"` registers the native backend for core's ops.
 *
 * TODO(ch05): point this at the-verdict.txt (or your own corpus) and enable once
 * GptModel.forward + crossEntropyLoss are implemented.
 */
function main(): void {
  const corpus = "language models learn from data one token at a time. ".repeat(50);
  const ids = gptEncode(corpus);

  const dataset = new LlmDataset(ids, 16, 8);
  const loader = new DataLoader(dataset, { batchSize: 4, shuffle: true });

  const model = new GptModel({
    ...GPT_CONFIG_124M,
    contextLength: 16,
    embDim: 64,
    numHeads: 4,
    numLayers: 2,
  });

  trainModel(model, loader, { epochs: 1, learningRate: 4e-4 });
}

main();
