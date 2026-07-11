"""Heuristic Query Intent Router.

Classifies natural language queries into specific retrieval intents and maps them
to the optimal LanceDB table routes to maximize search relevance and speed.
"""

import re
from typing import Literal, TypeAlias

from rfc_atlas.chunking.schema import LanceTableRoute

QueryIntent: TypeAlias = Literal[
    "conceptual_explanation",
    "protocol_history",
    "visual_structure",
    "syntax_grammar",
    "normative_query",
    "security_analysis",
]


class QueryRouter:
    """Classifies user queries to determine the optimal LanceDB table routes."""

    # Matches strict formatting, grammar structures, and protocol code definitions
    # to route queries directly to 'abnf' and 'sourcecode' tables.
    _SYNTAX_KEYWORDS = re.compile(
        r"\b(abnf|grammar|syntax|code|parse|format|rule|definition|header format)\b",
        re.IGNORECASE,
    )

    # Captures requests for spatial representations, state models, and visual flows
    # to prioritize the 'artwork' and 'table' LanceDB routes.
    _VISUAL_KEYWORDS = re.compile(
        r"\b(diagram|figure|artwork|table|flow|state machine|chart|visual)\b",
        re.IGNORECASE,
    )

    # Identifies threat modeling, encryption, and authorization queries
    # to ensure the dedicated 'security' considerations table is searched.
    _SECURITY_KEYWORDS = re.compile(
        r"\b(security|attack|vulnerabilit\w*|threat|mitigation|tls|crypto|encryption|oauth|auth\w*|hack)\b",
        re.IGNORECASE,
    )

    # Triggers on strict BCP-14 compliance language to ensure the retrieved
    # chunks contain rigid protocol requirements.
    _NORMATIVE_KEYWORDS = re.compile(
        r"\b(must|should|may|required|shall|mandatory|normative|strict|comply|compliance)\b",
        re.IGNORECASE,
    )

    # Detects queries about the lifecycle, obsoletion, or historical context
    # of a standard, bringing the 'references' table into scope alongside prose.
    _HISTORY_KEYWORDS = re.compile(
        r"\b(history|obsolet\w*|updat\w*|older|version|previous|deprecated|superseded)\b",
        re.IGNORECASE,
    )

    @classmethod
    def classify_intents(cls, query: str) -> set[QueryIntent]:
        """Classifies a raw text query into one or more operational intents.

        Args:
            query (str): The natural language query string.

        Returns:
            set[QueryIntent]: A set of matched intents. Defaults to
                conceptual_explanation if no specific heuristics trigger.
        """
        intents: set[QueryIntent] = set()

        if cls._SYNTAX_KEYWORDS.search(query):
            intents.add("syntax_grammar")
        if cls._VISUAL_KEYWORDS.search(query):
            intents.add("visual_structure")
        if cls._SECURITY_KEYWORDS.search(query):
            intents.add("security_analysis")
        if cls._NORMATIVE_KEYWORDS.search(query):
            intents.add("normative_query")
        if cls._HISTORY_KEYWORDS.search(query):
            intents.add("protocol_history")

        if not intents:
            intents.add("conceptual_explanation")

        return intents

    @classmethod
    def route_query(cls, query: str) -> list[LanceTableRoute]:
        """Maps a natural language query to the exact LanceDB tables required.

        Args:
            query (str): The natural language query string.

        Returns:
            list[LanceTableRoute]: A deduplicated, deterministically sorted list
                of tables to execute the hybrid search against.
        """
        intents = cls.classify_intents(query)
        tables: set[LanceTableRoute] = {"prose"}

        for intent in intents:
            if intent == "syntax_grammar":
                tables.update(["abnf", "sourcecode"])
            elif intent == "visual_structure":
                tables.update(["artwork", "table"])
            elif intent == "security_analysis":
                tables.add("security")
            elif intent == "protocol_history":
                tables.add("references")

        return sorted(tables)
