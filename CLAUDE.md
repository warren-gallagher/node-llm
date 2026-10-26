# CLAUDE.md

Companion code for the book [*Building an LLM in NodeJS*](https://book.humanexperiences.io): a GPT built from scratch in TypeScript on `@tensorflow/tfjs-node`. This repo is a fork, shared with a team that is working through the book together. See [README.md](README.md) for the package layout and how to run things.

## Branches

- `main` tracks the upstream repo. `fix` is the base of this fork: it holds the changes needed to install the repo and get the tests passing.
- Start new work on a branch off `fix`, not `main` (for example, `docs/reader-notes`).

## Commands

- `pnpm install`, `pnpm test`, `pnpm typecheck`, `pnpm check` (Biome lint and format check)
- Run a single lesson: `pnpm --filter @node-llm/chapters ch02:5` (scripts are in [packages/chapters/package.json](packages/chapters/package.json))
- Run a whole chapter: `pnpm ch02` … `pnpm ch05:train`

## Reader notes

[docs/reader-notes.md](docs/reader-notes.md) records questions and answers that came up while reading the book and running the demos. It is written for teammates and may be offered to the original author, so it must make sense to someone who wasn't part of the conversation.

After answering a question about the book, the code, or LLM concepts, decide whether it belongs in the notes:

- **Add it** if another reader is likely to ask the same thing: it connects demo values to real models, explains a design choice, clears up something confusing, or fills in background the book assumes.
- **Skip it** if it was about this machine's setup, a one-off debugging session, or something the book or code comments already explain well.
- **Update an existing entry** instead of adding a new one when the topic overlaps.

When it qualifies, propose the entry: say which chapter it goes under, give the heading, and say whether it is new or an update. Wait for the user to confirm before editing the file. Don't add entries silently.

How to write entries:

- Put them under the chapter heading that matches `packages/chapters/src/chNN-*`, in chapter order, and keep the Contents list current. Use a "General" section for topics that don't belong to one chapter.
- Use a question as the heading. Start with a **Context:** line linking to the relevant file or lines (relative links), then a **Short answer:**, then the details.
- Write in a neutral, explanatory voice, like a sidebar in the book. Don't refer to "our conversation", "you asked", or Claude.
- Mark facts about real models or tools that will change over time with "(as of YYYY-MM)". Only include numbers you are confident about, and say so when a value isn't published.
- When a diagram would make an explanation clearer (data flow, tensor shapes, how layers connect), embed it as a fenced `mermaid` code block, which GitHub renders. Don't add image files.
- Keep the notes separate from the upstream code: don't edit chapter files to add commentary. That keeps the notes easy to contribute upstream as one self-contained file.
