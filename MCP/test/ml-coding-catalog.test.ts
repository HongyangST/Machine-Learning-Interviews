import assert from "node:assert/strict";
import { test } from "node:test";

import { extractProblems } from "../src/catalog.js";

const SOURCE_PATH = "src/MLC/ml-coding.md";
const HEADER = `| Problem | Difficulty | Tags | Company tags | Answer | Interview focus |
| --- | --- | --- | --- | --- | --- |`;
const ROW =
  "| k-nearest neighbors | ![Medium](../assets/difficulty-medium.svg) | distance, ranking | Uber, LinkedIn, Meta | [Python](answer.py) | Pairwise distances and top-k selection |";

test("ML coding category headings preserve legacy problem IDs", () => {
  const legacy = extractProblems(
    SOURCE_PATH,
    `## Priority ML coding problems\n\n${HEADER}\n${ROW}`,
    "commit-a",
  );
  const categorized = extractProblems(
    SOURCE_PATH,
    `## Priority ML coding problems\n\n### Classic ML\n\n${HEADER}\n${ROW}`,
    "commit-a",
  );

  assert.equal(legacy.length, 1);
  assert.equal(categorized.length, 1);
  assert.equal(categorized[0].id, legacy[0].id);
  assert.equal(categorized[0].sourceHeading, "Priority ML coding problems");
  assert.equal(categorized[0].difficulty, "medium");
  assert.deepEqual(categorized[0].companies, ["LinkedIn", "Meta", "Uber"]);
  assert.ok(categorized[0].tags.includes("ranking"));
  assert.equal(categorized[0].prompt.includes("answer.py"), false);
});

