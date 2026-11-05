import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import type { GptConfig } from "@node-llm/core";
import { GptModel } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";
import { REPO_ROOT } from "./env";

/**
 * Int8 quantization of a saved checkpoint, for the credential demo's first derivation.
 *
 * Each tensor gets one scale, the largest absolute value divided by 127, and each value becomes
 * the nearest multiple of that scale, stored as a signed byte. Every step is defined exactly
 * (float32 scales, JavaScript's `Math.round`), so anyone holding the parent's weights can rerun
 * it and get the same bytes, which is what makes the derivation checkable.
 *
 * The quantized checkpoint holds `config.json` (the parent's, plus the method) and
 * `weights.q8.bin`: for each tensor in order, its scale as a little-endian float32, then its
 * values as int8.
 */

export const QUANTIZATION = {
  method: "symmetric int8, one scale per tensor",
  bits: 8,
  scale: "float32: the tensor's largest absolute value / 127",
  rounding: "Math.round of value / scale, clamped to [-127, 127]",
} as const;

export const QUANTIZED_WEIGHTS = "weights.q8.bin";

interface SavedConfig {
  config: GptConfig;
  shapes: number[][];
  quantization?: typeof QUANTIZATION;
}

function sizeOf(shape: number[]): number {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

/** The quantized bytes for float32 weights laid out tensor by tensor with these shapes. */
export function quantizeWeights(values: Float32Array, shapes: number[][]): Buffer {
  const total = shapes.reduce((sum, shape) => sum + 4 + sizeOf(shape), 0);
  const out = Buffer.alloc(total);
  let read = 0;
  let write = 0;
  for (const shape of shapes) {
    const size = sizeOf(shape);
    const tensor = values.subarray(read, read + size);
    let maxAbs = 0;
    for (const value of tensor) maxAbs = Math.max(maxAbs, Math.abs(value));
    const scale = Math.fround(maxAbs / 127);
    out.writeFloatLE(scale, write);
    write += 4;
    for (const value of tensor) {
      const q = scale === 0 ? 0 : Math.max(-127, Math.min(127, Math.round(value / scale)));
      out.writeInt8(q, write);
      write += 1;
    }
    read += size;
  }
  if (read !== values.length) {
    throw new Error(`the shapes cover ${read} values, but there are ${values.length}`);
  }
  return out;
}

/** The float32 weights the quantized bytes stand for: each value times its tensor's scale. */
export function dequantizeWeights(bytes: Uint8Array, shapes: number[][]): Float32Array {
  const buffer = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Float32Array(shapes.reduce((sum, shape) => sum + sizeOf(shape), 0));
  let read = 0;
  let write = 0;
  for (const shape of shapes) {
    const scale = buffer.readFloatLE(read);
    read += 4;
    for (let i = 0; i < sizeOf(shape); i++) {
      values[write++] = buffer.readInt8(read++) * scale;
    }
  }
  return values;
}

function checkpointDir(name: string): URL {
  return new URL(`checkpoints/${name}/`, REPO_ROOT);
}

/** Quantizes the checkpoint `parent` into a new checkpoint `name`. */
export function quantizeCheckpoint(parent: string, name: string): void {
  const from = checkpointDir(parent);
  const saved = JSON.parse(readFileSync(new URL("config.json", from), "utf8")) as SavedConfig;
  const bytes = readFileSync(new URL("weights.bin", from));
  const values = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);

  const to = checkpointDir(name);
  mkdirSync(to, { recursive: true });
  writeFileSync(new URL(QUANTIZED_WEIGHTS, to), quantizeWeights(values, saved.shapes));
  writeFileSync(
    new URL("config.json", to),
    JSON.stringify({ config: saved.config, shapes: saved.shapes, quantization: QUANTIZATION }),
  );
}

/** Whether a checkpoint holds quantized weights. */
export function isQuantized(name: string): boolean {
  const saved = JSON.parse(readFileSync(new URL("config.json", checkpointDir(name)), "utf8"));
  return saved.quantization !== undefined;
}

/** A model whose weights are the dequantized values of a quantized checkpoint. */
export function loadQuantized(name: string): GptModel {
  const dir = checkpointDir(name);
  const saved = JSON.parse(readFileSync(new URL("config.json", dir), "utf8")) as SavedConfig;
  const values = dequantizeWeights(readFileSync(new URL(QUANTIZED_WEIGHTS, dir)), saved.shapes);
  const model = new GptModel(saved.config);
  let offset = 0;
  for (const [i, w] of model.weights.entries()) {
    if (w.shape.join() !== saved.shapes[i]?.join()) {
      throw new Error(
        `checkpoint ${name}: weight ${i} has shape ${saved.shapes[i]}, expected ${w.shape}`,
      );
    }
    w.assign(tf.tensor(values.subarray(offset, offset + w.size), w.shape));
    offset += w.size;
  }
  return model;
}
