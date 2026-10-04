import { decode as gptDecode, encode as gptEncode } from "gpt-tokenizer";

/**
 * Chapter 2, Lesson 2: byte-pair encoding via `gpt-tokenizer`.
 *
 * We reuse the same BPE tokenizer GPT-2/3/4 use rather than building one from scratch.
 * `encode` maps text to token ids; `decode` maps ids back to the original text.
 */
function main(): void {
  const text = "I walk my dog everyday before the city wakes up";

  const ids = gptEncode(text);
  console.log("tokens:", ids);
  console.log("round-trip ok:", gptDecode(ids) === text);
}

main();
