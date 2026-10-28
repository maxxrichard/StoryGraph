"""Controlled vocabulary for genres, themes and moods.

The LLM tagger is free to emit any string, but labels are snapped to this taxonomy (with
synonym handling) before they reach the feature store so downstream features stay stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GENRES: tuple[str, ...] = (
    "literary_fiction",
    "crime",
    "thriller",
    "mystery",
    "horror",
    "science_fiction",
    "fantasy",
    "romance",
    "historical_fiction",
    "young_adult",
    "memoir",
    "biography",
    "history",
    "science",
    "technology",
    "business",
    "economics",
    "politics",
    "true_crime",
    "self_help",
    "travel",
    "nature",
    "philosophy",
    "religion",
    "comedy",
    "interview",
    "news",
    "sports",
    "health",
    "education",
)

THEMES: tuple[str, ...] = (
    "grief",
    "family",
    "isolation",
    "identity",
    "memory",
    "betrayal",
    "redemption",
    "survival",
    "coming_of_age",
    "power",
    "class",
    "money",
    "trust",
    "faith",
    "war",
    "migration",
    "nature",
    "technology",
    "justice",
    "love",
    "friendship",
    "ambition",
    "loss",
    "secrets",
    "the_sea",
    "community",
    "work",
    "madness",
    "the_supernatural",
    "fear",
    "coordination_failure",
    "institutions",
    "language",
    "home",
    "freedom",
)

MOODS: tuple[str, ...] = (
    "eerie",
    "melancholic",
    "tense",
    "hopeful",
    "warm",
    "wry",
    "urgent",
    "contemplative",
    "playful",
    "bleak",
    "cozy",
    "analytical",
    "suspenseful",
    "nostalgic",
    "angry",
    "serene",
)

SYNONYMS: dict[str, str] = {
    "sci-fi": "science_fiction",
    "scifi": "science_fiction",
    "sf": "science_fiction",
    "literary": "literary_fiction",
    "lit fic": "literary_fiction",
    "fiction": "literary_fiction",
    "detective": "crime",
    "whodunit": "mystery",
    "ghost story": "horror",
    "gothic": "horror",
    "autobiography": "memoir",
    "finance": "economics",
    "econ": "economics",
    "tech": "technology",
    "ocean": "the_sea",
    "sea": "the_sea",
    "maritime": "the_sea",
    "haunting": "the_supernatural",
    "ghosts": "the_supernatural",
    "supernatural": "the_supernatural",
    "bereavement": "grief",
    "mourning": "grief",
    "loneliness": "isolation",
    "solitude": "isolation",
    "bank run": "coordination_failure",
    "panic": "coordination_failure",
    "spooky": "eerie",
    "creepy": "eerie",
    "sad": "melancholic",
    "wistful": "nostalgic",
    "anxious": "tense",
    "funny": "playful",
    "humorous": "wry",
    "reflective": "contemplative",
    "thoughtful": "contemplative",
    "calm": "serene",
    "peaceful": "serene",
    "dark": "bleak",
    "gripping": "suspenseful",
    "comforting": "cozy",
    "comfortable": "cozy",
}


def _norm(label: str) -> str:
    return label.strip().lower().replace("-", " ").replace("/", " ")


@dataclass
class Taxonomy:
    genres: tuple[str, ...] = GENRES
    themes: tuple[str, ...] = THEMES
    moods: tuple[str, ...] = MOODS
    synonyms: dict[str, str] = field(default_factory=lambda: dict(SYNONYMS))

    def snap(self, label: str, field_name: str) -> str | None:
        """Map a free-text label to the taxonomy for ``field_name`` or return None."""
        vocab = getattr(self, field_name)
        raw = _norm(label)
        plain = label.strip().lower()
        candidates = [
            raw,
            raw.replace(" ", "_"),
            self.synonyms.get(plain, ""),
            self.synonyms.get(raw, ""),
        ]
        for c in candidates:
            if c and c in vocab:
                return c
        # loose match: taxonomy term contained in the label or vice versa (e.g. "crime drama" -> crime)
        for term in vocab:
            spaced = term.replace("_", " ")
            if spaced in raw or (len(raw) > 3 and raw in spaced):
                return term
        return None

    def prompt_block(self) -> str:
        return (
            f"GENRES: {', '.join(self.genres)}\n"
            f"THEMES: {', '.join(self.themes)}\n"
            f"MOODS: {', '.join(self.moods)}"
        )


DEFAULT_TAXONOMY = Taxonomy()
