import { describe, expect, it } from "vitest";
import { weightChangeProfile } from "../../src/credential-demo/forensics";
import { dequantizeWeights, quantizeWeights } from "../../src/credential-demo/quantize";
import { mulberry32 } from "../../src/credential-demo/seeded";

const shapes = [[8, 16], [16], [16, 4]];
const size = shapes.reduce((sum, shape) => sum + shape.reduce((a, b) => a * b, 1), 0);

function randomWeights(seed: number): Float32Array {
  const random = mulberry32(seed);
  return Float32Array.from({ length: size }, () => (random() - 0.5) * 0.2);
}

describe("weight-difference forensics", () => {
  const parent = randomWeights(1);

  it("sees identical weights as unchanged", () => {
    expect(weightChangeProfile(parent, parent.slice(), shapes)).toMatchObject({
      unchanged: 3,
      withinRounding: 0,
      beyondRounding: 0,
      largestRelativeChange: 0,
    });
  });

  it("sees quantization as changes within half a rounding step, in every tensor", () => {
    const quantized = dequantizeWeights(quantizeWeights(parent, shapes), shapes);
    const profile = weightChangeProfile(parent, quantized, shapes);
    expect(profile.beyondRounding).toBe(0);
    expect(profile.withinRounding).toBe(3);
  });

  it("sees training-like updates as changes beyond rounding", () => {
    const random = mulberry32(2);
    const trained = parent.map((value) => value + (random() - 0.5) * 0.02);
    const profile = weightChangeProfile(parent, trained, shapes);
    expect(profile.beyondRounding).toBe(3);
    expect(profile.largestRelativeChange).toBeGreaterThan(0.01);
  });

  it("notices one value moved by more than half a step", () => {
    const quantized = dequantizeWeights(quantizeWeights(parent, shapes), shapes);
    quantized[3] += 0.01;
    expect(weightChangeProfile(parent, quantized, shapes).beyondRounding).toBe(1);
  });

  it("refuses models of different sizes", () => {
    expect(() => weightChangeProfile(parent, parent.subarray(1), shapes)).toThrow("differ in size");
  });
});
