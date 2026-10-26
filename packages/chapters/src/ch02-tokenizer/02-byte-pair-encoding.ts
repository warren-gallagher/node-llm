import { decode as gptDecode, encode as gptEncode } from "gpt-tokenizer/encoding/r50k_base";

/**
 * Chapter 2, Lesson 2: byte-pair encoding via `gpt-tokenizer`.
 *
 * We reuse GPT-2's BPE tokenizer (`r50k_base`) rather than building one from scratch. The
 * package's default export is GPT-4o's `o200k_base`, whose ids don't fit a 50,257-token vocab.
 * `encode` maps text to token ids; `decode` maps ids back to the original text.
 */
function main(): void {
  const text = "I walk my dog everyday before the city wakes up";

  const ids = gptEncode(text);
  console.log("tokens:", ids);
  console.log("round-trip ok:", gptDecode(ids) === text);
}

main();
