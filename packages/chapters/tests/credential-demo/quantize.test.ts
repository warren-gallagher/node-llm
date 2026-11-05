import { describe, expect, it } from "vitest";
import { dequantizeWeights, quantizeWeights } from "../../src/credential-demo/quantize";

describe("int8 quantization", () => {
  const shapes = [[2, 3], [4], [2]];
  const values = new Float32Array([0.5, -1.27, 0.3, 0, 1.27, -0.01, 2, -2, 1, 0.004, 0, 0]);

  it("stores a float32 scale then int8 values for each tensor", () => {
    const bytes = quantizeWeights(values, shapes);
    expect(bytes.byteLength).toBe(3 * 4 + 12);
    expect(bytes.readFloatLE(0)).toBeCloseTo(0.01, 7);
    expect([...bytes.subarray(4, 10)].map((b) => (b << 24) >> 24)).toEqual([
      50, -127, 30, 0, 127, -1,
    ]);
  });

  it("keeps every value within half a step of the original", () => {
    const restored = dequantizeWeights(quantizeWeights(values, shapes), shapes);
    const scales = [Math.fround(1.27 / 127), Math.fround(2 / 127), 0];
    let i = 0;
    for (const [t, shape] of shapes.entries()) {
      for (let n = 0; n < shape.reduce((a, b) => a * b, 1); n++, i++) {
        expect(Math.abs(restored[i] - values[i])).toBeLessThanOrEqual(scales[t] / 2 + 1e-9);
      }
    }
  });

  it("maps an all-zero tensor to zeros with scale 0", () => {
    const bytes = quantizeWeights(values, shapes);
    expect(bytes.readFloatLE(4 + 6 + 4 + 4)).toBe(0);
    expect([...dequantizeWeights(bytes, shapes).subarray(10)]).toEqual([0, 0]);
  });

  it("gives the same bytes every time", () => {
    expect(quantizeWeights(values, shapes).equals(quantizeWeights(values.slice(), shapes))).toBe(
      true,
    );
  });

  it("refuses shapes that don't cover the values", () => {
    expect(() => quantizeWeights(values, [[2, 3]])).toThrow("the shapes cover 6 values");
  });
});
