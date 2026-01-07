from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class FeatureBuilder:
    input_mode: str
    lowercase_term_match: bool = True
    add_special_tokens_markers: bool = True
    marker_left: str = "[TGT]"
    marker_right: str = "[/TGT]"

    def build_text(self, term: str, context: Optional[str]) -> str:
        term = (term or "").strip()
        ctx = (context or "").strip()

        if self.input_mode == "context_only":
            return ctx if ctx else term

        if self.input_mode == "context_plus_term":
            if ctx:
                return f"Context: {ctx}\nTerm: {term}"
            return f"Term: {term}"

        if self.input_mode == "marked_context":
            if not ctx:
                # pas de contexte, fallback sur "Term: ..."
                return f"Term: {term}"
            return self._mark_term_in_context(term, ctx)

        raise ValueError(f"Unknown input_mode: {self.input_mode}")

    def _mark_term_in_context(self, term: str, context: str) -> str:
        """Entoure la première occurrence de `term` dans `context` avec des marqueurs.
        Si pas trouvé, on concatène context + term."""
        if not term:
            return context

        # Échapper pour regex
        t = re.escape(term)

        flags = re.IGNORECASE if self.lowercase_term_match else 0
        pattern = re.compile(rf"\b{t}\b", flags)

        def repl(match: re.Match) -> str:
            left = self.marker_left if self.add_special_tokens_markers else "<TGT>"
            right = self.marker_right if self.add_special_tokens_markers else "</TGT>"
            return f"{left} {match.group(0)} {right}"

        new_ctx, n = pattern.subn(repl, context, count=1)
        if n == 0:
            # si pas trouvé exactement, on fait un fallback moins strict (substring)
            if self.lowercase_term_match:
                idx = context.lower().find(term.lower())
            else:
                idx = context.find(term)

            if idx >= 0:
                left = self.marker_left if self.add_special_tokens_markers else "<TGT>"
                right = self.marker_right if self.add_special_tokens_markers else "</TGT>"
                new_ctx = context[:idx] + f"{left} " + context[idx:idx+len(term)] + f" {right}" + context[idx+len(term):]
                return new_ctx

            # sinon concat
            return f"{context}\nTerm: {term}"

        return new_ctx
