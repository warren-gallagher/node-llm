import { DataLoader, LlmDataset } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";
import { decode, encode } from "gpt-tokenizer";
import { describe, expect, it } from "vitest";

describe("chapters environment smoke test", () => {
  it("tfjs-node computes a matmul", () => {
    const a = tf.tensor2d([
      [1, 2],
      [3, 4],
    ]);
    const b = tf.tensor2d([
      [5, 6],
      [7, 8],
    ]);
    const c = tf.matMul(a, b);
    expect(Array.from(c.dataSync())).toEqual([19, 22, 43, 50]);
    tf.dispose([a, b, c]);
  });

  it("gpt-tokenizer round-trips text", () => {
    const text = "hello world, tokens!";
    expect(decode(encode(text))).toBe(text);
  });

  it("core data pipeline is importable across packages", () => {
    const ids = Array.from({ length: 40 }, (_, i) => i);
    const loader = new DataLoader(new LlmDataset(ids, 8, 4), { batchSize: 2 });
    expect([...loader][0]?.inputs[0]).toHaveLength(8);
  });
});
