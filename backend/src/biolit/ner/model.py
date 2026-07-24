from dataclasses import dataclass
from typing import Protocol

from biolit.config import Settings
from biolit.ner.windowing import predict_windowed

# BERT-family position embeddings cap the input at 512 tokens, and this checkpoint's
# tokenizer declares no model_max_length, so it never truncates -- an over-long document
# reaches the model and raises a size-mismatch RuntimeError. Real abstracts exceed it
# (2.2% of the BC5CDR test split; max observed 722 tokens), so text is windowed first.
# The budget leaves headroom for [CLS]/[SEP] and any tokenizer variation.
_MAX_WINDOW_TOKENS = 450


class Predictor(Protocol):
    def __call__(self, text: str) -> list[dict]: ...


@dataclass
class NerModel:
    predictor: Predictor

    @classmethod
    def load(cls, settings: Settings) -> "NerModel":
        # Heavy imports are lazy so the pure logic stays importable/testable offline.
        import torch
        from transformers import pipeline

        use_cuda = settings.ner_device in ("auto", "cuda") and torch.cuda.is_available()
        device = 0 if use_cuda else -1
        pipe = pipeline(
            "token-classification",
            model=settings.ner_model_id,
            aggregation_strategy="simple",
            device=device,
            batch_size=settings.ner_batch_size,
        )

        def count_tokens(chunk: str) -> int:
            return len(pipe.tokenizer(chunk)["input_ids"])  # type: ignore[union-attr,index]

        def predict(text: str) -> list[dict]:
            return predict_windowed(
                text,
                run=lambda chunk: list(pipe(chunk)),
                count_tokens=count_tokens,
                max_tokens=_MAX_WINDOW_TOKENS,
            )

        return cls(predictor=predict)
