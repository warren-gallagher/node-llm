import "@tensorflow/tfjs-backend-cpu";
// tfjs-core ships gradients separately; tfjs-node (used by the chapters) registers them itself.
import "@tensorflow/tfjs-core/dist/register_all_gradients";
import * as tf from "@tensorflow/tfjs-core";
import { beforeAll, describe, expect, it } from "vitest";
import type { GptConfig } from "../src/index";
import { FeedForward, GptModel, gelu, LayerNorm, TransformerBlock } from "../src/index";

const TINY: GptConfig = {
  vocabSize: 50,
  contextLength: 8,
  embDim: 16,
  numHeads: 4,
  numLayers: 2,
  dropout: 0.1,
  qkvBias: false,
};

beforeAll(async () => {
  await tf.setBackend("cpu");
  await tf.ready();
});

describe("LayerNorm", () => {
  it("gives each vector mean 0 and variance 1 over its features", () => {
    const norm = new LayerNorm(6);
    const x = tf.mul(tf.randomNormal([2, 3, 6]), 5);
    const y = norm.forward(x);
    const mean = tf.mean(y, -1);
    const variance = tf.mean(tf.square(tf.sub(y, tf.mean(y, -1, true))), -1);
    for (const m of mean.dataSync()) expect(m).toBeCloseTo(0, 5);
    for (const v of variance.dataSync()) expect(v).toBeCloseTo(1, 3);
    tf.dispose([x, y, mean, variance]);
    norm.dispose();
  });
});

describe("gelu", () => {
  it("matches the tanh approximation at known points", () => {
    const y = gelu(tf.tensor1d([-1, 0, 1, 3]));
    const [a, b, c, d] = Array.from(y.dataSync());
    expect(a).toBeCloseTo(-0.158808, 5);
    expect(b).toBe(0);
    expect(c).toBeCloseTo(0.841192, 5);
    expect(d).toBeCloseTo(2.996363, 5);
    y.dispose();
  });
});

describe("FeedForward and TransformerBlock", () => {
  it("preserve the [batch, seqLen, embDim] shape", () => {
    const ff = new FeedForward(16);
    const block = new TransformerBlock({ embDim: 16, numHeads: 4, contextLength: 8 });
    const x = tf.randomNormal<tf.Rank.R3>([2, 5, 16]);
    const fOut = ff.forward(x);
    const bOut = block.forward(x);
    expect(fOut.shape).toEqual([2, 5, 16]);
    expect(bOut.shape).toEqual([2, 5, 16]);
    tf.dispose([x, fOut, bOut]);
    ff.dispose();
    block.dispose();
  });
});

describe("GptModel", () => {
  it("maps token ids to logits of shape [batch, seqLen, vocabSize]", () => {
    const model = new GptModel(TINY);
    const ids = tf.tensor2d(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      undefined,
      "int32",
    );
    const logits = model.forward(ids);
    expect(logits.shape).toEqual([2, 3, TINY.vocabSize]);
    tf.dispose([ids, logits]);
    model.dispose();
  });

  it("is causal: changing a later token leaves earlier logits unchanged", () => {
    const model = new GptModel(TINY);
    const a = tf.tensor2d([[1, 2, 3, 4]], undefined, "int32");
    const b = tf.tensor2d([[1, 2, 3, 9]], undefined, "int32");
    const la = model.forward(a).arraySync()[0] as number[][];
    const lb = model.forward(b).arraySync()[0] as number[][];
    for (let pos = 0; pos < 3; pos++) {
      for (let v = 0; v < TINY.vocabSize; v++) {
        expect(lb[pos]?.[v]).toBeCloseTo(la[pos]?.[v] as number, 4);
      }
    }
    expect(lb[3]).not.toEqual(la[3]);
    tf.dispose([a, b]);
    model.dispose();
  });

  it("has the parameter count the architecture implies", () => {
    const { vocabSize: v, contextLength: c, embDim: d, numLayers: n } = TINY;
    const perBlock =
      2 * 2 * d + // two LayerNorms: scale + shift
      3 * d * d + // W_query, W_key, W_value (no qkv bias)
      (d * d + d) + // output projection + bias
      (d * 4 * d + 4 * d) + // feed-forward expand + bias
      (4 * d * d + d); // feed-forward contract + bias
    const expected = v * d + c * d + n * perBlock + 2 * d + d * v;

    const model = new GptModel(TINY);
    expect(model.numParameters()).toBe(expected);
    model.dispose();
  });

  it("gets a gradient for every weight", () => {
    const model = new GptModel(TINY);
    const ids = tf.tensor2d([[1, 2, 3, 4]], undefined, "int32");
    const { value, grads } = tf.variableGrads(
      () => tf.mean(model.forward(ids, true)) as tf.Scalar,
      model.weights,
    );
    expect(Object.keys(grads)).toHaveLength(model.weights.length);
    tf.dispose([ids, value, ...Object.values(grads)]);
    model.dispose();
  });
});
