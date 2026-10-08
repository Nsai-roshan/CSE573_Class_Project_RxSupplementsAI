
PRD: Multi-Agent RAG Platform with Automated Hallucination Evaluation
v2 extension of RxSupplementsAI (CSE 573 Group 21)
Base repo (fork this): KavJaggar/CSE573_Class_Project_RxSupplementsAI
Your fork: Nsai-roshan/<your-v2-repo-name>
Stack added in v2: Python, LangChain / LangGraph, Neo4j (wired in), FastAPI
Author: Sai Roshan Rao Nelavalli (team member, Group 21)
Status: Build spec — implementation via Codex, phase by phase

1. Goal
Extend the existing RxSupplementsAI supplement QA system into a **multi-agent
RAG platform**: every draft answer passes through a dedicated verification
agent that checks it against retrieved sources before it reaches the user, a
complexity router sends simple questions to a cheap model tier and hard ones
to a strong tier, and an automated eval harness scores groundedness and
citation accuracy as a CI gate on every commit.

The build must reproduce, by measurement, the results stated on the resume:

Multi-agent RAG answering from the existing 30,566-document supplement store.
Verification agent checking every draft against retrieved source material.
Hallucination rate reduced 40% against the measured v1 baseline.
Average query cost reduced 35% via complexity-based routing.
Eval harness scoring groundedness and citation accuracy on every commit.
2. What already exists (do not rebuild)
Inventory Codex must read first (Code/, Data/, Evaluation/):

Exists

Location

v2 reuses as

FAISS index + doc store (30,566 docs)

Data/CorpusData/natmed_documents.json, natmed_data.faiss

retrieval corpus, unchanged

BM25 index

Data/CorpusData/bm25_index.pkl, Code/BM25_corpus.py

keyword retrieval, unchanged

v1 query pipeline (Flask, FAISS top-3 + BM25 top-2 → Mistral via Ollama, streaming)

Code/query_phi.py

the frozen baseline — extract, don't modify

KG generation scripts + KG data

Code/create_kg.py, Code/KG_relationship_generation.py, Data/KnowledgeGraphData/

wire into v2 retriever (v1 leaves the KG block as an empty stub)

50 curated eval questions

Evaluation/evaluationquestions.txt

seed of the golden set

Batch test runner + prior results

Code/test_responses.py, Evaluation/TestResponses*.json

reference; superseded by eval/ harness

React frontend

Code/WebApp

untouched in v2

3. Non-goals
No changes to v1 behavior: Code/query_phi.py stays byte-identical as the
  measured baseline. All v2 code lives in Code/v2/.
No frontend changes in v2 (API-only).
No fine-tuning in v2. Model tiers are off-the-shelf (Ollama local or API).
No new corpus ingestion: the 30,566 supplement docs are the document store.
4. Success criteria
ID

Criterion

How it is measured

SC-1

≥40% relative reduction in hallucination rate vs v1 baseline

eval report: (H_v1 − H_v2) / H_v1 ≥ 0.40 on frozen golden set

SC-2

≥35% relative reduction in average cost per query vs v1

eval report: (C_v1 − C_v2) / C_v1 ≥ 0.35, tokens × price table

SC-3

100% of v2 answers carry citations; every served answer passed the verifier or the explicit abstention path

integration tests + served-answer log audit

SC-4

Eval harness runs in CI on every push; push fails if hallucination rate regresses >2 pts or citation precision drops >2 pts vs main

GitHub Actions workflow

SC-5

p95 end-to-end latency < 60s on the v2 pipeline (local models)

load script, 50 sequential queries

Definitions (frozen before any benchmark run):

Hallucination rate = fraction of answers containing ≥1 atomic claim not
  entailed by the retrieved source chunks, judged by the verifier model at
  temperature 0, with a 10% human spot-check.
Groundedness = supported atomic claims / total atomic claims.
Citation precision = citations whose cited chunk actually entails the
  attached claim / total citations emitted.
Citation recall = factual claims carrying ≥1 citation / total factual claims.
Cost per query = Σ over all LLM calls of (input_tokens × p_in +
  output_tokens × p_out) from Code/v2/config/prices.yaml. v1 is priced the
  same way, so the comparison is apples-to-apples even on local models.
5. Architecture (v2)
 EXISTING (frozen)                    NEW IN v2
 Data/CorpusData ──┐
                   ├─▶ Code/v2/retrieval.py ── FAISS top-8 + BM25 top-8,
 Data/KnowledgeGraphData ─┘           RRF fusion → top-6, + Neo4j 1-hop facts
                                              │
User question ──▶ Code/v2/router.py ──▶ SIMPLE|COMPLEX → Tier A | Tier B
   (cheap classifier LLM)                     │
                                              ▼
                              Code/v2/drafter.py ── answer + [doc:chunk]
                              (tier-selected LLM)    citations + claim list
                                              │
                                              ▼
                              Code/v2/verifier.py ── per-claim entailment
                              (strong tier, temp 0)    vs cited chunks
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼ all supported                 ▼ unsupported found             ▼ still failing
     DELIVER + citations              regenerate (≤2, stricter            ABSTAIN: supported
                                      grounding prompt)                  claims only + notice
                                              │
                              Code/v2/api.py (FastAPI): POST /ask,
                              GET /health, GET /metrics
                              Code/v2/costing.py: per-call token ledger
All v2 agents are LangGraph nodes sharing a typed AnswerState
(question, tier, chunks, draft, claims, verdicts, cost ledger, latency).
v1's Flask app is untouched; v2 serves on FastAPI (default :8182).

6. Functional requirements
6.1 Phase-0 groundwork
FR-0.1: Fork the base repo to your account. All v2 code under Code/v2/;
  v1 files byte-identical.
FR-0.2: Replace the hardcoded Ollama LAN IP in v1-derived code with
  OLLAMA_BASE_URL env (default http://localhost:11434); .env.example.
FR-0.3: Code/v2/config/ — models.yaml (tier_a, tier_b, router_model,
  verifier_model, LLM_PROVIDER=ollama|openai), prices.yaml (per-1K-token
  in/out prices), eval.yaml (golden path, CI subset 20, thresholds 2.0).
6.2 Baseline extraction (Code/v2/baseline.py)
FR-2.1: Extract v1's exact retrieval+generation behavior
  (FAISS top-3 + BM25 top-2, single Mistral call, citations requested) into a
  callable baseline_answer(question) with no behavior change.
FR-2.2: Log tokens/cost per call via Code/v2/costing.py
  (tiktoken counts × prices.yaml).
6.3 Retrieval agent (Code/v2/retrieval.py)
FR-3.1: Widen to FAISS top-8 + BM25 top-8 → reciprocal-rank fusion → top-6.
FR-3.2: Inventory Code/create_kg.py, Code/KG_relationship_generation.py,
  Data/KnowledgeGraphData/; wire the KG into retrieval — entity-link the
  question, pull 1-hop Neo4j neighborhood facts, append as kg: context.
  (v1's query_phi.py has an empty #KG RETRIEVAL stub; v2 fills it for real.)
FR-3.3: Degrade gracefully if Neo4j is unreachable (log warning, dense-only).
FR-3.4: Return chunk IDs, text, doc titles, scores for provenance.
6.4 Query router (Code/v2/router.py)
FR-4.1: Few-shot classifier (cheap model, temp 0) → SIMPLE (single-fact
  lookup, definition, dosage) or COMPLEX (interactions, multi-hop, comparison,
  safety synthesis); log label + confidence.
FR-4.2: SIMPLE → Tier A, COMPLEX → Tier B per models.yaml.
  Defaults (local): A = mistral:7b, B = llama3.1:70b; provider switch needs
  no code change.
FR-4.3: Router accuracy ≥80% against complexity labels in the golden set
  (ablation in eval report).
6.5 Drafter (Code/v2/drafter.py)
FR-5.1: Tier-selected generation; every factual sentence ends with ≥1
  citation in [doc_id:chunk_id] format (v1's citation style, now enforced).
FR-5.2: Emit structured claim list: [{claim, citation_ids[]}].
FR-5.3: Record input/output tokens per call into the run's cost ledger.
FR-5.4: Keep the medical-safety behavior: if retrieved context is
  irrelevant, say so instead of answering (v1's "tell the user you do not
  know", now machine-checked).
6.6 Verifier (Code/v2/verifier.py)
FR-6.1: Split draft into atomic claims (LLM, temp 0, Pydantic JSON schema);
  entailment-check each against its cited chunks (strong tier, temp 0) →
  {claim_id, verdict: SUPPORTED|UNSUPPORTED|PARTIAL, evidence_chunk_ids}.
FR-6.2: Any UNSUPPORTED → regenerate with failing claims listed + stricter
  grounding system prompt; max 2 regenerations.
FR-6.3: Still failing after 2 → abstain: return supported claims only with
  a "could not fully verify" notice; log the failure class. Never silently
  serve an unverified claim as verified.
FR-6.4: Verifier–human agreement ≥85% on a 20-question spot-check, recorded
  in Code/v2/eval/reports/verifier-agreement.md.
6.7 Eval harness (Code/v2/eval/)
FR-7.1: Golden set Code/v2/eval/golden.jsonl — 80 questions: the
  existing 50 from Evaluation/evaluationquestions.txt (human-written answer
  keys + supporting doc IDs added) + 30 new, each
  `{id, question, expected_answer, supporting_doc_ids[], complexity,
  tags[]}`. Frozen by SHA-256; harness refuses mismatches without a version bump.
FR-7.2: Code/v2/eval/run.py --mode v1|v2 --limit N → per-question answers,
  writing Code/v2/eval/reports/<ts>/report.json + report.md: hallucination
  rate, groundedness, citation precision/recall, avg $/query, router accuracy,
  p50/p95 latency.
FR-7.3: Code/v2/eval/compare.py diffs two reports → SC-1/SC-2 verdicts.
FR-7.4: 3 runs per mode, same seeds; report means. Judge prompts versioned
  in Code/v2/eval/prompts/, temp 0.
6.8 Serving API (Code/v2/api.py)
FR-8.1: FastAPI POST /ask {question} →
  `{answer, citations[], claims[{text, verdict, chunk_ids}], model_tier,
  cost_usd, latency_ms, verified, abstained}`.
FR-8.2: GET /health, GET /metrics (queries, abstentions, avg cost,
  verifier reject rate).
FR-8.3: README documents setup from fork clone to first /ask in ≤15 min
  (existing index files are reused, not rebuilt).
6.9 CI gate (.github/workflows/v2-eval.yml)
FR-9.1: On push: install, run pytest, run harness on a fixed 20-question CI
  subset in v2 mode.
FR-9.2: Commit Code/v2/eval/baseline-ci.json; fail the push if
  hallucination rate regresses >2 pts or citation precision drops >2 pts.
7. Benchmark protocol (how SC-1 and SC-2 are earned)
Human-verifies the 80-question answer key; freeze golden.jsonl, record hash.
Run --mode v1 3× → H0, C0. 3. Run --mode v2 3× → H1, C1.
Publish Code/v2/eval/reports/<date>/BENCHMARK.md: raw numbers, model
   versions, price table, seeds, SC-1/SC-2 verdicts.
README + resume numbers must match this report exactly.
8. Repo layout (v2 additions)
Code/v2/
├── config/          # models.yaml, prices.yaml, eval.yaml
├── retrieval.py     # fused FAISS+BM25+KG retrieval
├── router.py        # complexity classifier + tier selection
├── drafter.py       # tier-selected generation + claim list
├── verifier.py      # claim decomposition + entailment checks
├── graph.py         # LangGraph wiring: router→retriever→drafter→verifier
├── pipeline.py      # answer(question) -> AnswerResult
├── baseline.py      # extracted v1 behavior (frozen)
├── costing.py       # token ledger × prices.yaml
├── api.py           # FastAPI
├── eval/            # golden.jsonl, prompts/, run.py, compare.py,
│                    # baseline-ci.json, reports/
└── tests/           # unit + integration
9. Non-functional requirements
NFR-1: Full 80-question eval (local models) completes in < 90 min.
NFR-2: No secrets in code; .env for keys, .env.example committed.
NFR-3: Unit tests for router/verifier/costing; 4 end-to-end tests
  (simple → Tier A verified; complex → Tier B verified; off-topic →
  abstention; adversarial unanswerable → no confident cited answer).
NFR-4: v1 files byte-identical (CI check: git diff on Code/query_phi.py
  etc. must be empty).
10. Risks and mitigations
Risk

Mitigation

70B local model too heavy for dev

Ollama quantized default; price-table costing keeps benchmarks valid; API tiers for the final frozen run only

Verifier judge bias

strong-tier judge + 10% human spot-check + agreement report

KG scripts bit-rotted

Phase 0 inventories them; v2 degrades to dense-only if KG unusable, noted in README

Golden-set leakage into tuning

freeze hash before benchmark; any prompt change after freeze → re-run both modes

11. Glossary
Atomic claim: one verifiable factual statement extracted from a draft.
v1 / v2: the original Group 21 pipeline vs this multi-agent extension.
Tier A / Tier B: cheap vs strong LLM tier selected by the router.
Abstention: refusing to serve unverified claims after 2 failed regens.
