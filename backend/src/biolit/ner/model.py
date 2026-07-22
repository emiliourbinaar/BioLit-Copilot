from dataclasses import dataclass
from typing import Protocol

from biolit.config import Settings


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

        def predict(text: str) -> list[dict]:
            if not text or not text.strip():
                return []
            return list(pipe(text))

        return cls(predictor=predict)
