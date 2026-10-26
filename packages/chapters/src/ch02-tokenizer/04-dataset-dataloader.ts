import { DataLoader, LlmDataset } from "@node-llm/core";
import "@tensorflow/tfjs-node";
import { encode as gptEncode } from "gpt-tokenizer/encoding/r50k_base";

/**
 * Chapter 2, Lesson 4: Dataset & DataLoader.
 *
 * `LlmDataset` turns a token stream into sliding-window (input, target) samples, and
 * `DataLoader` batches and shuffles them. With `asTensors: true` it yields int32 tensors
 * of shape [batchSize, seqLen], ready to feed the model.
 *
 * Importing `@tensorflow/tfjs-node` registers the backend that tensor creation needs.
 */
function main(): void {
  const ids = gptEncode("I walk my dog everyday before the city wakes up and the streets are still calm");

  const dataset = new LlmDataset(ids, 4, 4);
  console.log(`dataset samples: ${dataset.length}`);

  // Plain-array batches (the default).
  const arrayLoader = new DataLoader(dataset, { batchSize: 2, shuffle: false });
  for (const batch of arrayLoader) {
    console.log("array batch shape:", `${batch.inputs.length}x${batch.inputs[0]?.length}`);
    break;
  }

  // Tensor batches: int32 tensors of shape [batchSize, seqLen].
  const tensorLoader = new DataLoader(dataset, { batchSize: 2, shuffle: false, asTensors: true });
  for (const batch of tensorLoader) {
    console.log("tensor batch shape:", batch.inputs.shape, batch.inputs.dtype);
    // Tensors are not garbage-collected, so dispose once done.
    batch.inputs.dispose();
    batch.targets.dispose();
    break;
  }
}

main();
