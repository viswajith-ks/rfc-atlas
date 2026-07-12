import pytest

from rfc_atlas.retrieval.query_router import QueryRouter


@pytest.mark.parametrize(
    ("query", "expected_intents"),
    [
        (
            "Explain how the TCP handshake works.",
            {"conceptual_explanation"},
        ),
        (
            "What is the ABNF grammar for the HTTP Host header?",
            {"syntax_grammar"},
        ),
        (
            "Show me the state machine diagram for TLS.",
            {"visual_structure", "security_analysis"},
        ),
        (
            "Are there any security vulnerabilities or threats in OAuth?",
            {"security_analysis"},
        ),
        (
            "When MUST a server reject the packet?",
            {"normative_query"},
        ),
        (
            "Why was RFC 2616 obsoleted and what is its history?",
            {"protocol_history"},
        ),
        (
            "Show me the grammar syntax and the security diagram.",
            {"syntax_grammar", "security_analysis", "visual_structure"},
        ),
    ],
)
def test_classify_intents(query: str, expected_intents: set[str]) -> None:
    intents = QueryRouter.classify_intents(query)
    assert intents == expected_intents


@pytest.mark.parametrize(
    ("query", "expected_tables"),
    [
        ("Explain how TCP works.", ["prose"]),
        ("What is the ABNF grammar?", ["abnf", "prose", "sourcecode"]),
        ("Show me the diagram.", ["artwork", "prose", "table"]),
        ("What are the security threats?", ["prose", "security"]),
        ("When MUST it happen?", ["abnf", "prose"]),
        ("What is the history of this obsolete protocol?", ["prose", "references"]),
        (
            "Show the ABNF grammar and the security diagram.",
            ["abnf", "artwork", "prose", "security", "sourcecode", "table"],
        ),
    ],
)
def test_route_query_deterministic_sorting(
    query: str, expected_tables: list[str]
) -> None:
    tables = QueryRouter.route_query(query)
    assert tables == expected_tables
    assert tables == sorted(tables), (
        "CRITICAL: Router output is not deterministically sorted!"
    )
