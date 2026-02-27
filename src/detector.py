from dataclasses import dataclass

from transformers import pipeline


@dataclass(frozen=True)
class DetectionResult:
    probability_ai: float
    label: str


class AiDetector:
    def __init__(self, model_name: str):
        self._classifier = pipeline("text-classification", model=model_name, tokenizer=model_name)

    def detect(self, text: str) -> DetectionResult:
        # Truncate long posts for model limits while preserving practical performance.
        result = self._classifier(text, truncation=True, max_length=512)[0]
        label = str(result["label"]).lower()
        score = float(result["score"])

        if "human" in label:
            probability_ai = 1.0 - score
        elif "ai" in label or "machine" in label:
            probability_ai = score
        else:
            # Fallback for unknown label conventions.
            probability_ai = score

        return DetectionResult(probability_ai=max(0.0, min(1.0, probability_ai)), label=label)
