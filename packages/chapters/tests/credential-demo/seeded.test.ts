import { afterEach, describe, expect, it } from "vitest";
import { mulberry32, seedMathRandom } from "../../src/credential-demo/seeded";

function take(next: () => number, count: number): number[] {
  return Array.from({ length: count }, () => next());
}

describe("credential demo seeding", () => {
  const original = Math.random;
  afterEach(() => {
    Math.random = original;
  });

  it("gives the same sequence for the same seed", () => {
    expect(take(mulberry32(42), 5)).toEqual(take(mulberry32(42), 5));
    expect(take(mulberry32(42), 5)).not.toEqual(take(mulberry32(43), 5));
  });

  it("returns floats in [0, 1)", () => {
    for (const value of take(mulberry32(7), 1000)) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it("replaces Math.random and restores it", () => {
    const restore = seedMathRandom(42);
    const seeded = take(Math.random, 3);
    restore();
    expect(Math.random).toBe(original);
    expect(seeded).toEqual(take(mulberry32(42), 3));
  });
});
