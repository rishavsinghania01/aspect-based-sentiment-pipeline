from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass(frozen=True)
class AspectMention:
    aspect: str
    matched_term: str
    sentiment: float
    context: str


def load_aspect_lexicon(path: str | Path | None = None) -> dict[str, list[str]]:
    if path is None:
        payload = files(__package__).joinpath("default_aspects.json").read_text(encoding="utf-8")
    else:
        payload = Path(path).read_text(encoding="utf-8")

    raw = json.loads(payload)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("The aspect lexicon must be a non-empty JSON object.")

    lexicon: dict[str, list[str]] = {}
    for aspect, aliases in raw.items():
        if not isinstance(aspect, str) or not aspect.strip():
            raise ValueError("Every aspect name must be a non-empty string.")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(f"Aspect '{aspect}' must contain at least one alias.")
        cleaned = sorted({str(alias).strip().lower() for alias in aliases if str(alias).strip()})
        if not cleaned:
            raise ValueError(f"Aspect '{aspect}' has no usable aliases.")
        lexicon[aspect.strip()] = cleaned
    return lexicon


class ClauseAspectExtractor:
    """Attach sentiment to an aspect using the local clause around its mention."""

    _CLAUSE_BREAK = re.compile(
        r"(?:[.!?;]+|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\byet\b|\bwhereas\b)",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        aspect_lexicon: dict[str, list[str]] | None = None,
        sentiment_analyser: SentimentIntensityAnalyzer | None = None,
    ) -> None:
        self.aspect_lexicon = aspect_lexicon or load_aspect_lexicon()
        self.sentiment_analyser = sentiment_analyser or SentimentIntensityAnalyzer()
        self._patterns = {
            aspect: re.compile(
                r"\b(?:"
                + "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
                + r")\b",
                flags=re.IGNORECASE,
            )
            for aspect, aliases in self.aspect_lexicon.items()
        }

    def extract(self, text: str) -> list[AspectMention]:
        mentions: list[AspectMention] = []
        for clause in self._split_clauses(text):
            matches = self._find_matches(clause)
            for index, (start, end, aspect, matched_term) in enumerate(matches):
                previous_end = matches[index - 1][1] if index else 0
                next_start = matches[index + 1][0] if index + 1 < len(matches) else len(clause)
                left = (previous_end + start) // 2 if index else 0
                right = (end + next_start) // 2 if index + 1 < len(matches) else len(clause)
                context = clause[left:right].strip(" ,:-") or clause.strip()
                sentiment = self.sentiment_analyser.polarity_scores(context)["compound"]
                mentions.append(
                    AspectMention(
                        aspect=aspect,
                        matched_term=matched_term.lower(),
                        sentiment=float(sentiment),
                        context=context,
                    )
                )
        return mentions

    def _split_clauses(self, text: str) -> list[str]:
        normalised = re.sub(r"\s+", " ", str(text)).strip()
        primary = [part.strip() for part in self._CLAUSE_BREAK.split(normalised) if part.strip()]
        clauses: list[str] = []
        for part in primary:
            fragments = [
                fragment.strip()
                for fragment in re.split(r"(?:,|\band\b|\bor\b)", part, flags=re.IGNORECASE)
                if fragment.strip()
            ]
            fragments_with_aspects = sum(bool(self._find_matches(item)) for item in fragments)
            if len(fragments) > 1 and fragments_with_aspects > 1:
                clauses.extend(fragments)
            else:
                clauses.append(part)
        return clauses

    def _find_matches(self, clause: str) -> list[tuple[int, int, str, str]]:
        candidates: list[tuple[int, int, str, str]] = []
        for aspect, pattern in self._patterns.items():
            for match in pattern.finditer(clause):
                candidates.append((match.start(), match.end(), aspect, match.group(0)))

        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        selected: list[tuple[int, int, str, str]] = []
        for candidate in candidates:
            start, end, _, _ = candidate
            if any(
                start < chosen_end and end > chosen_start
                for chosen_start, chosen_end, *_ in selected
            ):
                continue
            selected.append(candidate)
        return selected
