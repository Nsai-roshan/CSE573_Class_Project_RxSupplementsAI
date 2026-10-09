# v1 Baseline Notes

Phase 1 freezes the v1 retrieval and prompt behavior for measurement:

- FAISS retrieves the top 3 documents.
- BM25 retrieves the top 2 passages.
- One generation call is made with the v1 prompt and source-name citation style.
- The v1 source used Mistral through Ollama, but that Ollama endpoint is unavailable
  in this environment.
- For comparable measurements, both baseline and v2 use the configured provider's
  Tier B model. This isolates the measured delta to the multi-agent architecture,
  not a model change.
- Baseline evaluation returns non-streaming text and records one token-cost ledger
  entry for the generation call.

