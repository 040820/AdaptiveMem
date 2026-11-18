# MRAgent baseline on LoCoMo conv-26

This snapshot runs the original graph construction and QA with Qwen3-30B-A3B and BGE-M3 through OpenAI-compatible endpoints. The launcher reads the root .env file. Keep it local.

Set LICO_LLM_BASE_URL (port 8001, path /v1), LICO_LLM_MODEL=qwen3-30b-a3b, LICO_LLM_API_KEY, LICO_EMBEDDING_BASE_URL (port 8015, path /v1), LICO_EMBEDDING_MODEL=bge-m3, LICO_EMBEDDING_API_KEY, and LICO_EMBEDDING_DIMENSIONS=1024. Run launch.py --preflight-only to check the connection, then run launch.py to build or reuse artifacts and answer the questions. A completed result is reused. Run verify.py to verify the bundled results.

The baseline contains 199 predictions, including 60 ERROR rows. Non-adversarial token F1 was 0.241749. The LLM judge was not run. See metrics.json for per-category results. Raw logs and the LongMemEval LFS placeholder were left out of this export.
