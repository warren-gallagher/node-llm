import { LlmDataset } from "@node-llm/core";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgpu";
import * as tf from "@tensorflow/tfjs-core";

const output = document.getElementById("output") as HTMLPreElement;

function log(line: string): void {
  output.textContent += `\n${line}`;
}

async function main(): Promise<void> {
  output.textContent = "initializing…";

  // Prefer WebGPU; fall back to the pure-JS CPU backend.
  let backend = "webgpu";
  try {
    await tf.setBackend("webgpu");
    await tf.ready();
  } catch {
    backend = "cpu";
    await tf.setBackend("cpu");
    await tf.ready();
  }
  log(`backend: ${tf.getBackend()} (requested ${backend})`);

  // The exact same tfjs-core ops that run under tfjs-node in the chapters run here on GPU.
  const a = tf.tensor2d([
    [1, 2],
    [3, 4],
  ]);
  const b = tf.tensor2d([
    [5, 6],
    [7, 8],
  ]);
  const c = tf.matMul(a, b);
  log(`matmul result: ${await c.data()}`);
  tf.dispose([a, b, c]);

  // Prove the shared core library is usable in the browser too.
  const dataset = new LlmDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4, 2);
  log(`LlmDataset samples from @node-llm/core: ${dataset.length}`);
}

main().catch((err) => log(`error: ${err}`));
