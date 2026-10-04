import "@tensorflow/tfjs-backend-cpu";
import * as tf from "@tensorflow/tfjs-core";
import { beforeAll, describe, expect, it } from "vitest";

/**
 * Proves the backend-agnostic design: core uses tfjs-core ops, and a consumer registers a
 * backend (here the pure-JS CPU backend) for them to execute against.
 */
describe("tfjs-core with a registered backend", () => {
  beforeAll(async () => {
    await tf.setBackend("cpu");
    await tf.ready();
  });

  it("runs a matmul", () => {
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
});
