# QLoRA notebook: run on Kaggle (free T4)

1. Create a Kaggle notebook, then go to **Settings → Accelerator → GPU T4** (turn on **Internet** too).
2. **File → Import Notebook** and upload `finetune_qlora.ipynb`.
3. **Run All** (about 30–60 min). It prints sample generations and the base vs. fine-tuned ROUGE-L.
4. Download `qlora_adapter.zip` from the **Output** panel (`/kaggle/working`).
5. Serve it with `serve_finetuned.ipynb` (below). `LLM_PROVIDER=local` is no longer supported by the backend.

# Serving notebook: `serve_finetuned.ipynb` (Kaggle T4, Internet ON)

1. Import `serve_finetuned.ipynb` into a new Kaggle notebook and attach `qlora_adapter.zip` with **Add Input**, or upload it with the widget in cell 2.
2. **Run All**. vLLM serves Qwen2.5-1.5B + the adapter as model `scholarpulse`; if vLLM fails, a transformers + peft fallback serves the same API.
3. Copy the printed `FINETUNED_BASE_URL=<url>/v1` and `FINETUNED_API_KEY=<key>` lines into `research_assistant/.env`, then restart the backend.
4. The test cell should print a cited TLDR. Answers to PDF questions then show the "Fine-tuned Qwen-1.5B (Kaggle GPU) · Xs" badge (check `GET /health/finetuned`).
5. Keep the last cell running. The URL is public, but every request needs the API key (a new URL and key each session), so don't share the key or save the notebook's output publicly.
