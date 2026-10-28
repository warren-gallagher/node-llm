import "@tensorflow/tfjs-backend-cpu";
// tfjs-core ships gradients separately; tfjs-node (used by the chapters) registers them itself.
import "@tensorflow/tfjs-core/dist/register_all_gradients";
import * as tf from "@tensorflow/tfjs-core";
import { beforeAll, describe, expect, it } from "vitest";
import type { GptConfig } from "../src/index";
import {
  ClassificationDataset,
  ClassificationLoader,
  classificationAccuracy,
  classificationLoss,
  GptClassifier,
  GptModel,
  trainClassifier,
} from "../src/index";

const TINY: GptConfig = {
  vocabSize: 20,
  contextLength: 8,
  embDim: 16,
  numHeads: 2,
  numLayers: 2,
  dropout: 0,
  qkvBias: true,
};

beforeAll(async () => {
  await tf.setBackend("cpu");
  await tf.ready();
});

describe("ClassificationDataset", () => {
  it("pads every text to the longest one", () => {
    const data = new ClassificationDataset([[1, 2, 3], [4], [5, 6]], [0, 1, 0]);
    expect(data.maxLength).toBe(3);
    expect(data.at(1)).toEqual({ input: [4, 50256, 50256], label: 1 });
    expect(data.at(2).input).toEqual([5, 6, 50256]);
  });

  it("truncates texts longer than a given maxLength", () => {
    const data = new ClassificationDataset([[1, 2, 3, 4], [5]], [1, 0], 2, 0);
    expect(data.at(0).input).toEqual([1, 2]);
    expect(data.at(1).input).toEqual([5, 0]);
  });

  it("rejects a label count that does not match the texts", () => {
    expect(() => new ClassificationDataset([[1], [2]], [0])).toThrow();
  });
});

describe("ClassificationLoader", () => {
  const data = new ClassificationDataset(
    Array.from({ length: 10 }, (_, i) => [i]),
    Array.from({ length: 10 }, (_, i) => i % 2),
  );

  it("counts batches with and without the last partial batch", () => {
    expect(new ClassificationLoader(data, { batchSize: 4 }).length).toBe(3);
    expect(new ClassificationLoader(data, { batchSize: 4, dropLast: true }).length).toBe(2);
  });

  it("yields inputs and labels in order when not shuffled", () => {
    const [first] = new ClassificationLoader(data, { batchSize: 3 });
    expect(first).toEqual({ inputs: [[0], [1], [2]], labels: [0, 1, 0] });
  });
});

describe("GptClassifier", () => {
  it("outputs one score per class at every position", () => {
    const model = new GptClassifier(new GptModel(TINY), 2);
    const ids = tf.tensor2d([[1, 2, 3, 4, 5]], undefined, "int32");
    const all = model.forward(ids);
    const last = model.lastTokenLogits(ids);
    expect(all.shape).toEqual([1, 5, 2]);
    expect(last.shape).toEqual([1, 2]);
    expect(Array.from(last.dataSync())).toEqual(Array.from(all.dataSync().slice(8, 10)));
    tf.dispose([ids, all, last]);
    model.dispose();
  });

  it("trains only the last block, the final LayerNorm and the new head", () => {
    const base = new GptModel(TINY);
    const model = new GptClassifier(base, 3);
    const size = (ws: tf.Variable[]) => ws.reduce((sum, w) => sum + w.size, 0);
    const expected = size(base.blocks[1]?.weights ?? []) + 2 * TINY.embDim + (TINY.embDim * 3 + 3);
    expect(size(model.trainableWeights)).toBe(expected);
    model.dispose();
  });
});

describe("classificationLoss", () => {
  it("is the cross-entropy of the last token's scores against each label", () => {
    const model = new GptClassifier(new GptModel(TINY), 2);
    const inputs = tf.tensor2d(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      undefined,
      "int32",
    );
    const labels = tf.tensor1d([1, 0], "int32");

    const logProbs = tf.logSoftmax(model.lastTokenLogits(inputs), -1).arraySync() as number[][];
    const expected = -((logProbs[0]?.[1] as number) + (logProbs[1]?.[0] as number)) / 2;

    const loss = classificationLoss(model, inputs, labels);
    expect(loss.dataSync()[0]).toBeCloseTo(expected, 5);
    tf.dispose([inputs, labels, loss]);
    model.dispose();
  });
});

describe("trainClassifier", () => {
  // Label 1 when the text ends in token 7, else 0.
  const texts = Array.from({ length: 64 }, (_, i) => [1 + (i % 5), 2 + (i % 3), i % 2 ? 7 : 9]);
  const labels = texts.map((t) => (t[2] === 7 ? 1 : 0));
  const data = new ClassificationDataset(texts, labels);

  it("learns a simple rule and leaves frozen weights unchanged", () => {
    const base = new GptModel(TINY);
    const model = new GptClassifier(base, 2);
    const loader = new ClassificationLoader(data, { batchSize: 8, shuffle: true, dropLast: true });
    const frozen = base.weights[0] as tf.Variable; // the token embedding
    const frozenBefore = frozen.dataSync().slice();
    const headBefore = (model.headWeights[0] as tf.Variable).dataSync().slice();

    const history = trainClassifier(model, loader, {
      epochs: 15,
      learningRate: 0.01,
      weightDecay: 0.1,
      evalEvery: 10,
      valLoader: new ClassificationLoader(data, { batchSize: 8 }),
    });

    expect(history.accuracies).toHaveLength(15);
    expect(history.examplesSeen).toBe(15 * 64);
    expect(classificationAccuracy(model, new ClassificationLoader(data, { batchSize: 8 }))).toBe(1);
    expect(Array.from(frozen.dataSync())).toEqual(Array.from(frozenBefore));
    expect(Array.from((model.headWeights[0] as tf.Variable).dataSync())).not.toEqual(
      Array.from(headBefore),
    );
    model.dispose();
  });
});
