import { decode as gptDecode, encode as gptEncode } from "gpt-tokenizer/encoding/r50k_base";

/**
 * Chapter 2, Lesson 3: data sampling with a sliding window.
 *
 * Each training pair is (input, target), where the target is the input shifted right
 * by one, so every position's label is exactly the next token.
 */
function main(): void {
  const ids = gptEncode("I walk my dog everyday before the city wakes up");

  const contextSize = 4;
  const input = ids.slice(0, contextSize);
  const target = ids.slice(1, contextSize + 1);

  console.log("input: ", input);
  console.log("target:", target);

  // The same idea shown as a growing context predicting the next token.
  for (let i = 1; i <= contextSize; i++) {
    const context = ids.slice(0, i);
    const next = ids[i];
    if (next === undefined) break;
    console.log(`${gptDecode(context)} ---> ${gptDecode([next])}`);
  }
}

main();
