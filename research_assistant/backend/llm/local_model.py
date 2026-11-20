"""
local_model.py
--------------
Local chat model: Qwen2.5-1.5B-Instruct + the QLoRA adapter trained in
``notebooks/finetune_qlora.ipynb``, run on CPU in full precision (no 4-bit;
bitsandbytes needs a GPU). The adapter is merged into the base weights at
load time for faster generation. Expect several seconds per sentence on CPU.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.config import get_settings

log = logging.getLogger(__name__)


class LocalLLM:
    """Base model with the fine-tuned LoRA adapter merged in.

    Args:
        base_model: HuggingFace ID of the base model. Defaults to ``settings.local_base_model``.
        adapter_dir: Directory holding ``adapter_config.json`` and the adapter weights.
            Defaults to ``settings.local_adapter_dir``.

    Raises:
        FileNotFoundError: If the adapter directory has no ``adapter_config.json``.
    """

    def __init__(self, base_model: str | None = None, adapter_dir: str | Path | None = None) -> None:
        settings = get_settings()
        self.base_model: str = base_model or settings.local_base_model
        self.adapter_dir: Path = Path(adapter_dir or settings.local_adapter_dir)
        if not (self.adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No QLoRA adapter in {self.adapter_dir}. Train it with "
                "notebooks/finetune_qlora.ipynb and unzip qlora_adapter.zip there."
            )
        log.info("Loading %s + adapter %s on CPU", self.base_model, self.adapter_dir)
        tok_src = self.adapter_dir if (self.adapter_dir / "tokenizer_config.json").exists() else self.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        base = AutoModelForCausalLM.from_pretrained(self.base_model, dtype=torch.float32)
        self.model = PeftModel.from_pretrained(base, str(self.adapter_dir)).merge_and_unload().eval()

    @torch.no_grad()
    def generate(
        self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        """Generate the assistant reply for a chat.

        Args:
            messages: OpenAI-style ``[{"role": ..., "content": ...}]`` messages.
            max_tokens: Maximum new tokens.
            temperature: 0 for greedy decoding, otherwise sampling temperature.

        Returns:
            The generated reply text.
        """
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt")
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        out = self.model.generate(
            **enc, max_new_tokens=max_tokens, pad_token_id=self.tokenizer.pad_token_id, **sampling
        )
        return self.tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
