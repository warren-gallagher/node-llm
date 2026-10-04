/**
 * Chapter 2, Lesson 1: a simple word-level tokenizer.
 *
 * Splits text into words/punctuation, assigns each unique token a stable integer id,
 * and supports encode/decode. This is the warm-up before byte-pair encoding.
 */
function tokenize(text: string): string[] {
  return text.match(/\w+|[^\w\s]/g) ?? [];
}

function main(): void {
  const text = "I walk my dog, everyday.";
  const tokens = tokenize(text);

  // Build a vocabulary: each unique token gets a stable integer id.
  const vocab = new Map<string, number>();
  for (const token of tokens) {
    if (!vocab.has(token)) vocab.set(token, vocab.size);
  }
  const idToToken = [...vocab.keys()];

  const encode = (t: string): number[] => tokenize(t).map((tok) => vocab.get(tok) ?? -1);
  const decode = (ids: number[]): string => ids.map((id) => idToToken[id] ?? "").join(" ");

  const ids = encode(text);
  console.log("tokens:", tokens);
  console.log("vocab size:", vocab.size);
  console.log("ids:", ids);
  console.log("decoded:", decode(ids));
}

main();
