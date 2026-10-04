import "@tensorflow/tfjs-backend-cpu";
import { describe, expect, it } from "vitest";
import { DataLoader, LlmDataset } from "../src/index";

describe("data pipeline", () => {
  it("builds sliding-window (input, target) samples", () => {
    const ids = Array.from({ length: 40 }, (_, i) => i);
    const dataset = new LlmDataset(ids, 8, 4);
    const sample = dataset.at(0);
    expect(sample.input).toHaveLength(8);
    expect(sample.target).toHaveLength(8);
    // target is input shifted right by one
    expect(sample.target[0]).toBe((sample.input[0] ?? -1) + 1);
  });

  it("batches via the dataloader", () => {
    const ids = Array.from({ length: 40 }, (_, i) => i);
    const dataset = new LlmDataset(ids, 8, 4);
    const loader = new DataLoader(dataset, { batchSize: 2, shuffle: false });

    const first = [...loader][0];
    expect(first).toBeDefined();
    expect(first?.inputs).toHaveLength(2);
    expect(first?.inputs[0]).toHaveLength(8);
  });

  it("yields int32 tensor batches when asTensors is set", () => {
    const ids = Array.from({ length: 40 }, (_, i) => i);
    const dataset = new LlmDataset(ids, 8, 4);
    const loader = new DataLoader(dataset, { batchSize: 2, asTensors: true });

    // Pull only the first batch so we don't allocate tensors for the whole dataset.
    const first = loader[Symbol.iterator]().next().value;
    expect(first).toBeDefined();
    if (!first) return;

    expect(first.inputs.shape).toEqual([2, 8]);
    expect(first.inputs.dtype).toBe("int32");

    // Tensors are not garbage-collected, so free them once asserted.
    first.inputs.dispose();
    first.targets.dispose();
  });
});
