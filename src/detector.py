from dataclasses import dataclass

from transformers import pipeline


@dataclass(frozen=True)
class DetectionResult:
    probability_ai: float
    label: str


class AiDetector:
    def __init__(self, model_name: str):
        self._classifier = pipeline("text-classification", model=model_name, tokenizer=model_name)
        raw_id2label = getattr(getattr(self._classifier, "model", None), "config", None)
        raw_id2label = getattr(raw_id2label, "id2label", {}) or {}
        self._id2label: dict[int, str] = {}
        for key, value in raw_id2label.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            self._id2label[idx] = str(value).lower()

    def _semantic_label(self, label: str) -> str:
        lowered = label.lower()
        if lowered.startswith("label_"):
            try:
                idx = int(lowered.split("_", 1)[1])
            except (IndexError, ValueError):
                return lowered
            mapped = self._id2label.get(idx)
            if mapped:
                return mapped
        return lowered

    def detect(self, text: str) -> DetectionResult:
        # Truncate long posts for model limits while preserving practical performance.
        results = self._classifier(text, truncation=True, max_length=512, top_k=None)
        if not isinstance(results, list) or not results:
            raise RuntimeError("Detector returned no classification results")

        top = max(results, key=lambda item: float(item["score"]))
        top_label = str(top["label"]).lower()
        top_score = float(top["score"])

        ai_score = None
        human_score = None
        for item in results:
            raw_label = str(item["label"])
            semantic = self._semantic_label(raw_label)
            score = float(item["score"])
            if "human" in semantic:
                human_score = score
            elif "ai" in semantic or "machine" in semantic or "generated" in semantic:
                ai_score = score

        if ai_score is not None:
            probability_ai = ai_score
        elif human_score is not None:
            probability_ai = 1.0 - human_score
        else:
            # Fallback for unknown label conventions.
            probability_ai = top_score

        return DetectionResult(
            probability_ai=max(0.0, min(1.0, probability_ai)),
            label=self._semantic_label(top_label),
        )
