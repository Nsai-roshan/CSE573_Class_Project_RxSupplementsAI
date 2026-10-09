# Golden set notes

`golden.jsonl` contains the 50 inherited evaluation questions (`v1-01` through
`v1-50`) and 30 additional questions (`v2-01` through `v2-30`). Every record
starts with `human_verified: false`; this phase deliberately does not freeze
the answer key.

Tagging uses `single-fact`, `definition`, and `dosage` for simple lookups;
`interaction`, `comparison`, `multi-hop`, and `safety` for questions requiring
several facts or a risk synthesis; and `unanswerable` when the supplement
corpus cannot answer the question. A question may have multiple tags.

Before changing `human_verified` to `true`, a human verifier must check every
item against the corpus: (1) the expected answer is medically and semantically
consistent with the cited evidence, (2) every `supporting_doc_ids` value is a
real Phase-2 `doc_id:chunk_index` identifier whose text supports the answer,
(3) the complexity label matches the router definition, (4) tags describe the
question's required reasoning and include `unanswerable` when appropriate, and
(5) safety-sensitive answers do not imply diagnosis, individualized dosing, or
unsafe treatment. Empty supporting IDs are required for unanswerable items.

After all 80 records pass that review, update the dataset hash and version in
`config/eval.yaml` in the same commit. The harness refuses a hash mismatch.
