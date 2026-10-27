import "@tensorflow/tfjs-backend-cpu";
// tfjs-core ships gradients separately; tfjs-node (used by the chapters) registers them itself.
import "@tensorflow/tfjs-core/dist/register_all_gradients";
import * as tf from "@tensorflow/tfjs-core";
import { beforeAll, describe, expect, it } from "vitest";
import type { GptConfig } from "../src/index";
import {
  crossEntropyLoss,
  DataLoader,
  evaluateLoss,
  GptModel,
  LlmDataset,
  trainModel,
} from "../src/index";

const TINY: GptConfig = {
  vocabSize: 20,
  contextLength: 8,
  embDim: 16,
  numHeads: 2,
  numLayers: 1,
  dropout: 0,
  qkvBias: false,
};

beforeAll(async () => {
  await tf.setBackend("cpu");
  await tf.ready();
});

describe("crossEntropyLoss", () => {
  it("equals the mean negative log-probability of the target tokens", () => {
    const model = new GptModel(TINY);
    const inputs = tf.tensor2d(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      undefined,
      "int32",
    );
    const targets = tf.tensor2d(
      [
        [2, 3, 4],
        [5, 6, 7],
      ],
      undefined,
      "int32",
    );

    const logProbs = tf.logSoftmax(model.forward(inputs), -1).arraySync() as number[][][];
    const t = targets.arraySync();
    const picked = t.flatMap((row, b) => row.map((id, pos) => logProbs[b]?.[pos]?.[id] as number));
    const expected = -picked.reduce((sum, v) => sum + v, 0) / picked.length;

    const loss = crossEntropyLoss(model, inputs, targets);
    expect(loss.dataSync()[0]).toBeCloseTo(expected, 4);
    tf.dispose([inputs, targets, loss]);
    model.dispose();
  });
});

describe("GptModel.generate", () => {
  it("greedy decoding appends maxNewTokens tokens and is deterministic", () => {
    const model = new GptModel(TINY);
    const prompt = tf.tensor2d([[1, 2]], undefined, "int32");
    const a = model.generate(prompt, 5);
    const b = model.generate(prompt, 5);
    expect(a.shape).toEqual([1, 7]);
    expect(a.arraySync()).toEqual(b.arraySync());
    expect(a.arraySync()[0]?.slice(0, 2)).toEqual([1, 2]);
    tf.dispose([prompt, a, b]);
    model.dispose();
  });

  it("keeps working past the context length by cropping to the last tokens", () => {
    const model = new GptModel(TINY);
    const prompt = tf.tensor2d([[1, 2, 3]], undefined, "int32");
    const out = model.generate(prompt, TINY.contextLength + 4);
    expect(out.shape).toEqual([1, 3 + TINY.contextLength + 4]);
    tf.dispose([prompt, out]);
    model.dispose();
  });

  it("with top-k = 1, sampling at any temperature matches greedy", () => {
    const model = new GptModel(TINY);
    const prompt = tf.tensor2d([[3, 1]], undefined, "int32");
    const greedy = model.generate(prompt, 6);
    const sampled = model.generate(prompt, 6, { temperature: 5, topK: 1, seed: 42 });
    expect(sampled.arraySync()).toEqual(greedy.arraySync());
    tf.dispose([prompt, greedy, sampled]);
    model.dispose();
  });

  it("stops when the end token is produced, without appending it", () => {
    const model = new GptModel(TINY);
    const prompt = tf.tensor2d([[1, 2]], undefined, "int32");
    const first = model.generate(prompt, 1);
    const eosId = first.arraySync()[0]?.[2] as number;
    const out = model.generate(prompt, 5, { eosId });
    expect(out.arraySync()).toEqual([[1, 2]]);
    tf.dispose([prompt, first, out]);
    model.dispose();
  });
});

describe("trainModel", () => {
  it("lowers the loss on a repeating sequence", () => {
    const ids = Array.from({ length: 200 }, (_, i) => i % 10);
    const loader = new DataLoader(new LlmDataset(ids, 8, 4), { batchSize: 4 });
    const model = new GptModel(TINY);

    const before = evaluateLoss(model, loader);
    const history = trainModel(model, loader, {
      epochs: 3,
      learningRate: 0.01,
      weightDecay: 0.1,
      evalEvery: 5,
    });
    const after = evaluateLoss(model, loader);

    expect(history.length).toBeGreaterThan(0);
    expect(after).toBeLessThan(before / 2);
    model.dispose();
  });
});
