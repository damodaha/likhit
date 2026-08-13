"""Tests for the THIRD Latin-side veto: the document-scope acronym axis (VOL-180).

The two shipped vetoes (`27d74f0`'s :func:`_reads_as_latin_text`, `5084fb8`'s
:func:`_reads_as_latin_words`) both judge a run on its **own** text. What they cannot
reach is a run that is *nothing but a bare acronym* — v13 renders `QOC` (3 chars,
twice) and `ECOD ` (5 chars) as Devanagari that spells nothing, precisely because
there is no surrounding context in any of the three to be judged on. The evidence has
to come from outside the run, at document scope.

**Why the shape alone cannot be the rule, which is what these tests mostly guard.**
Corpus-wide (`runs/vol180/strict-calibration-635286f0.json`) 7,864 remapped runs hold
a short all-caps ASCII token that both shipped vetoes miss — 41x the whole of
`27d74f0`. Vetoing on that shape would be a licence to stop decoding wherever two
capitals appear. It is only the candidate generator; the document-scope survivor
condition is what cuts 7,864 to **16 fires, 16/16 genuine English, 0 Nepali touched**.

**The keystroke fragments below are the calibration's real failure mode, not
decoration.** A first pass tokenized on a punctuation class and produced 37 fires of
which **21 were spurious** — every one a Preeti keystroke fragment cut out of the
middle of a keystroke word, because the tokenizer split on legacy symbols: `6L` out of
`w/f}6L` (धरौटी), `G6L` out of `Uof/]G6L` (ग्यारेन्टी), `OG` out of `OG;]kmnfOl6;`
(इन्सेफलाइटिस). That is 43% precision on a reading, not the 92% the automatic
`false_positive` flag reported — a *short* keystroke run has fewer than two dictionary
words in its decode, so "≥ 2 dictionary words" is a sound definition of *provably*
Nepali and a weak detector of *actually* Nepali.
"""

from __future__ import annotations

from likhit.extractors.font_based import (
    _ACRONYM_EDGE,
    _ACRONYM_FORBIDDEN,
    LegacyMapChoice,
    _acronym_tokens,
    _content_legacy_veto_flags,
    _decodes_as_legacy_devanagari,
    _is_wellformed_devanagari,
    _reads_as_latin_text,
)
from likhit.extractors.legacy_maps import get_converter_for_map

SPINS = get_converter_for_map("Spins")
SPINS_CHOICE = {"Spins": LegacyMapChoice(map_key="Spins", validity=None)}


def _span(font: str, text: str) -> dict[str, str]:
    return {"font": font, "text": text}


# Genuine acronyms, verbatim from the 16 fires that were read individually
# (runs/vol180/strict-calibration-635286f0.json). `ECOD` and `QOC` are VOL-126's own
# targets — the residue this axis exists to reach.
GENUINE_ACRONYMS = ["MIS", "IEE", "DPR", "PLGSP", "ECOD", "QOC"]

# The spurious class, verbatim from the loose pass's 21 fires. NONE of these may
# yield a qualifying token: they are keystroke words, and a token cut out of one is an
# artefact of the tokenizer rather than an acronym.
KEYSTROKE_WORDS = [
    "w/f}6L",  # धरौटी      -- loose tokenizer gave `6L`
    "Uof/]G6L",  # ग्यारेन्टी  -- gave `G6L`
    ";]G6/",  # सेन्टर      -- gave `G6`
    "8]en]kd]G6",  # डेभलेपमेन्ट -- gave `G6`
    "OG;]kmnfOl6;",  # इन्सेफलाइटिस -- gave `OG`
    "x'G5,",  # हुन्छ       -- gave `G5`
    "8f6f PG6«L",  # डाटा एन्ट्री -- gave `PG6`
]


def test_genuine_acronyms_qualify() -> None:
    for token in GENUINE_ACRONYMS:
        assert _acronym_tokens(token) == frozenset({token}), token
    # ...and are found inside real English context, which is where `QOC`'s own
    # survivor evidence lives (`Quality Of Care, QOC` on page 231).
    assert _acronym_tokens("Quality Of Care, QOC") == frozenset({"QOC"})


def test_no_keystroke_word_yields_a_qualifying_token() -> None:
    """The 21 spurious fires, and the defect that produced them.

    Whitespace delimitation plus `_ACRONYM_FORBIDDEN` is what kills these. The loose
    tokenizer `[A-Za-z0-9/&().,:;+\\-]+` split them at legacy symbols and handed back
    the fragment as though it were a word.
    """

    for word in KEYSTROKE_WORDS:
        assert _acronym_tokens(word) == frozenset(), word


def test_note_1_two_uppercase_LETTERS_not_letters_or_digits() -> None:
    """`36L` is घटी: a real whitespace-delimited all-caps ASCII keystroke word.

    It is the one spurious fire that is NOT a tokenizer artefact, and it is why the
    rule needs `>= 2 uppercase letters` rather than the issue's sketch of "2-5
    characters, all of them uppercase ASCII letters **or digits**". Under the sketch
    `36L` qualifies on one letter.
    """

    assert _acronym_tokens("36L") == frozenset()
    assert _acronym_tokens("3A") == frozenset()
    assert _acronym_tokens("12") == frozenset()
    # Two letters is enough, with or without digits alongside.
    assert _acronym_tokens("ID2") == frozenset({"ID2"})
    assert _acronym_tokens("AB") == frozenset({"AB"})


def test_note_2_an_edge_strip_can_never_expose_a_forbidden_character() -> None:
    """The module asserts the two sets are disjoint at import; this pins that."""

    assert not (_ACRONYM_FORBIDDEN & frozenset(_ACRONYM_EDGE))
    assert _acronym_tokens(";]G6/") == frozenset()
    assert _acronym_tokens(";]G6/.") == frozenset()
    # Ordinary English edge punctuation IS stripped, which is the point of the class.
    assert _acronym_tokens("(QOC),") == frozenset({"QOC"})
    assert _acronym_tokens('"MIS";') == frozenset({"MIS"})


def test_the_forbidden_set_is_subsumed_by_the_shape_test() -> None:
    """§8 states `_ACRONYM_FORBIDDEN` as an independent condition. It is not.

    Every character in it is already excluded by "ASCII uppercase or digit", so the
    membership test cannot currently reject a token the shape test accepts — deleting
    it changes no outcome, which a mutation run confirmed. It is kept because it is
    the spec's wording and becomes load-bearing if the shape test is ever relaxed to
    admit lowercase or symbols; this test is what makes that relaxation visible
    instead of silent.
    """

    for char in _ACRONYM_FORBIDDEN:
        assert not (("A" <= char <= "Z") or ("0" <= char <= "9")), char


def test_whitespace_delimitation_is_what_excludes_three_fragment_shapes() -> None:
    """The rule that is actually load-bearing against the spurious class.

    `G6L`, `OG` and `PG6` PASS every shape condition — 2-5 chars, all uppercase ASCII
    or digits, two uppercase letters. Nothing about their shape disqualifies them.
    They are excluded only because they are *parts* of a whitespace-delimited
    keystroke word, which is why the tokenizer splits on whitespace and nothing else.
    """

    for fragment in ("G6L", "OG", "PG6"):
        assert _acronym_tokens(fragment) == frozenset({fragment}), (
            f"{fragment} is shape-legal; only whitespace delimitation excludes it"
        )
    # In the words they were cut out of, they are unreachable.
    assert _acronym_tokens("Uof/]G6L") == frozenset()
    assert _acronym_tokens("OG;]kmnfOl6;") == frozenset()
    assert _acronym_tokens("8f6f PG6«L") == frozenset()
    # The other four are excluded by note 1's two-letter floor instead.
    for fragment in ("6L", "G6", "G5", "36L"):
        assert _acronym_tokens(fragment) == frozenset(), fragment


def test_length_bounds() -> None:
    assert _acronym_tokens("A") == frozenset()
    assert _acronym_tokens("ABCDEF") == frozenset()
    assert _acronym_tokens("ABCDE") == frozenset({"ABCDE"})


def test_the_axis_vetoes_a_bare_acronym_run_when_the_document_attests_it() -> None:
    """The whole point: a 3-character run no other axis can judge.

    `QOC` alone is far below `_reads_as_latin_text`'s 16-character floor and holds no
    dictionary word, so both shipped vetoes decline it and v13 remaps it into
    Devanagari that spells nothing.
    """

    spans = [_span("Spins", "QOC")]
    assert _reads_as_latin_text("QOC", SPINS("QOC")) is False, "axis 1 declines it"
    assert _content_legacy_veto_flags(spans, SPINS_CHOICE) == [False]
    assert _content_legacy_veto_flags(spans, SPINS_CHOICE, frozenset({"QOC"})) == [True]


def test_without_survivor_evidence_the_axis_declines() -> None:
    """The shape is only the candidate generator — 7,864 runs carry it."""

    spans = [_span("Spins", "QOC")]
    assert _content_legacy_veto_flags(spans, SPINS_CHOICE, frozenset({"MIS"})) == [
        False
    ]
    assert _content_legacy_veto_flags(spans, SPINS_CHOICE, frozenset()) == [False]


def test_an_empty_survivor_set_is_exactly_the_pre_VOL180_behaviour() -> None:
    """The regression guard that matters for the generation build.

    Every caller that does not pass a survivor set must get byte-identical decisions
    to the two-axis veto. If this ever diverges, a tree built on this branch would
    differ from v13 for reasons unrelated to the acronym axis.
    """

    runs = [
        "QOC",
        "ECOD ",
        "Random rubble stone masonry work with 1:4",
        "w/f}6L",
        ";]G6/ 8]en]kd]G6",
        "MIS",
        "",
        "   ",
    ]
    for text in runs:
        spans = [_span("Spins", text)]
        assert _content_legacy_veto_flags(
            spans, SPINS_CHOICE, frozenset()
        ) == _content_legacy_veto_flags(spans, SPINS_CHOICE), text


def test_the_axis_runs_second_and_cannot_override_axis_1() -> None:
    """§8 requires a SECOND pass, decided only on runs axis 1 has declined.

    Both are one-sided (each only ever declines to remap), so ordering cannot change a
    veto into a remap — but it can change *which* axis is credited, and `QOC`'s own
    survivor evidence is created by axis 1 firing on `Quality Of Care, QOC`. A run
    axis 1 already vetoes must stay vetoed whatever the survivor set says.
    """

    english = "Random rubble stone masonry work with 1:4"
    spans = [_span("Spins", english)]
    assert _reads_as_latin_text(english, SPINS(english)) is True
    for survivors in (frozenset(), frozenset({"MIS"}), frozenset({"QOC"})):
        assert _content_legacy_veto_flags(spans, SPINS_CHOICE, survivors) == [True]


def test_the_veto_decides_a_whole_same_font_run_not_one_span() -> None:
    """Same run unit as the other two axes: a vetoed run is kept whole."""

    spans = [_span("Spins", "EC"), _span("Spins", "OD"), _span("Spins", " ")]
    assert _content_legacy_veto_flags(spans, SPINS_CHOICE, frozenset({"ECOD"})) == [
        True,
        True,
        True,
    ]


def test_a_non_candidate_font_is_never_flagged() -> None:
    spans = [_span("Helvetica", "QOC")]
    assert _content_legacy_veto_flags(spans, SPINS_CHOICE, frozenset({"QOC"})) == [
        False
    ]


# ---------------------------------------------------------------------------
# VOL-212: SURVIVOR PURITY. Latin *shape* is not evidence of Latin.
#
# The 13 tests above guard the shape test and the veto's scope. None of them can
# see this defect, because all of them hand the veto a survivor set built by
# hand: the defect is in how `detect_latin_acronym_survivors` BUILDS that set.
# ---------------------------------------------------------------------------

# Verbatim from `runs/vol197/fire-sweep-caps-token-f7071d15.json`, fire 26:
# `pdfs/report_annual-report/11129__n30-Annual Report 2071.pdf`, page 265, font Spins.
VOL212_RUN = (
    "ljj/0fx? oyfy+ b]lvg] cj:yf 5}g . ;fy}, nul;6df ljdfgsf] "
    "rf]s Og 6fOd, rf]s ckm 6fOd, PG6L "
)
# That document's ENTIRE survivor vocabulary under the shipped axis. No English at
# all -- three Preeti keystroke fragments, which is the defect in one line.
VOL212_IMPURE_VOCABULARY = {"OG6": "इन्ट", "P06L": "एण्टी", "PG6L": "एन्टी"}
# The 8 distinct tokens carrying the 25 genuine fires of the same sweep, across the
# other 10 firing documents. `runs/vol212/FINDING-discriminator-a3f21c8e.md`.
VOL212_GENUINE = {
    "MIS": ":क्ष्क्",
    "NS": "ल्क्",
    "GI": "न्क्ष्",
    "QOC": "त्तइऋ",
    "DPR": "म्एच्",
    "ECOD": "भ्ऋइम्",
    "HDPE": "ज्म्एभ्",
    "IEE": "क्ष्भ्भ्",
}


def test_the_impure_vocabulary_is_latin_SHAPED_which_is_why_note_3_missed_it() -> None:
    """`PG6L` passes every condition VOL-180 §8 imposes. That is the whole bug."""

    for token in VOL212_IMPURE_VOCABULARY:
        assert _acronym_tokens(token) == frozenset({token}), token
        assert sum(1 for char in token if "A" <= char <= "Z") >= 2, token


def test_11129_whole_survivor_vocabulary_is_impure() -> None:
    """Acceptance #1: `PG6L`, `P06L` and `OG6` must not qualify as survivors."""

    for token, decoded in VOL212_IMPURE_VOCABULARY.items():
        assert SPINS(token) == decoded, token
        assert _decodes_as_legacy_devanagari(token, SPINS_CHOICE), token


def test_11129_run_decodes_once_the_vocabulary_is_purified() -> None:
    """Acceptance #1, at the veto: fire 26 disappears and the Nepali comes back.

    The assertion that matters is the last one — published v12 and v13 both hold
    `चोक इन टाइम` for this run and none of the raw keystrokes, so the shipped axis was
    replacing fluent Nepali with ASCII.
    """

    spans = [_span("Spins", VOL212_RUN)]
    impure = frozenset(VOL212_IMPURE_VOCABULARY)
    assert _reads_as_latin_text(VOL212_RUN, SPINS(VOL212_RUN)) is False, (
        "axis 1 declines"
    )
    assert _content_legacy_veto_flags(spans, SPINS_CHOICE, impure) == [True], "fire 26"

    purified = frozenset(
        token
        for token in impure
        if not _decodes_as_legacy_devanagari(token, SPINS_CHOICE)
    )
    assert purified == frozenset(), "nothing in this document attests English"
    assert _content_legacy_veto_flags(spans, SPINS_CHOICE, purified) == [False]
    assert "चोक इन टाइम" in SPINS(VOL212_RUN)


def test_the_25_genuine_recoveries_survive_purification() -> None:
    """Acceptance #2, at token level: a narrowing that kills these is not a fix."""

    for token, decoded in VOL212_GENUINE.items():
        assert SPINS(token) == decoded, token
        assert not _decodes_as_legacy_devanagari(token, SPINS_CHOICE), token


def test_C4_alone_carries_QOC_so_it_cannot_be_dropped() -> None:
    """Which of the six conditions is load-bearing, measured not assumed.

    C3 (halant-final) carries 7 of the 8 genuine tokens. `QOC` is the exception: its
    decode `त्तइऋ` ends in a vowel letter, so only C4 (a non-initial independent vowel,
    which Nepali writes as a matra) keeps it. Drop C4 and `QOC`'s 2 fires die.
    """

    assert not SPINS("QOC").endswith("्"), "C3 does not fire on QOC"
    assert not _is_wellformed_devanagari(SPINS("QOC")), "C4 does"
    for token in ("MIS", "NS", "GI", "DPR", "ECOD", "HDPE", "IEE"):
        assert SPINS(token).endswith("्"), f"C3 carries {token}"


def test_the_predicate_accepts_real_nepali_or_the_filter_is_inert() -> None:
    """The other side of the one-sidedness: it must not call everything malformed."""

    for word in ("नेपाल", "सरकार", "कार्यालय", "काठमाडौं", "विरुद्ध", "मिति", "डॉलर"):
        assert _is_wellformed_devanagari(word), word


def test_each_of_the_six_conditions_rejects_its_own_shape() -> None:
    assert not _is_wellformed_devanagari(":क्ष्क"), "C1 non-Devanagari"
    assert not _is_wellformed_devanagari("ँब्त्त"), "C2 initial combining mark"
    assert not _is_wellformed_devanagari("ल्क्"), "C3 halant-final"
    assert not _is_wellformed_devanagari("त्तइऋ"), "C4 non-initial independent vowel"
    assert not _is_wellformed_devanagari("त्ी"), "C5 vowel sign after halant"
    assert not _is_wellformed_devanagari("ध्ब्ीी"), "C6 two vowel signs in a row"
    assert not _is_wellformed_devanagari(""), "empty is not a word"
    assert not _is_wellformed_devanagari("MIS"), (
        "an untouched ASCII token is not Nepali"
    )


def test_purity_is_ANY_candidate_map_not_ALL() -> None:
    """`DAX` (11115's vocabulary) is the corpus case that separates the two.

    Spins reads it as `म्ब्ह्` — halant-final, so malformed — while Kantipur reads it
    as `म्ब्हृ`, a well-formed word shape. A document carrying both candidate maps must
    not use it as evidence of English: one map reading it as Nepali is enough.
    """

    spins_only = {"Spins": LegacyMapChoice(map_key="Spins", validity=None)}
    kantipur_only = {"K": LegacyMapChoice(map_key="Kantipur", validity=None)}
    both = {**spins_only, **kantipur_only}
    assert not _decodes_as_legacy_devanagari("DAX", spins_only)
    assert _decodes_as_legacy_devanagari("DAX", kantipur_only)
    assert _decodes_as_legacy_devanagari("DAX", both), "any(), not all()"


def test_a_document_with_no_candidate_map_has_no_purity_opinion() -> None:
    assert not _decodes_as_legacy_devanagari("PG6L", {})
    assert not _decodes_as_legacy_devanagari(
        "PG6L", {"Spins": LegacyMapChoice(map_key=None, validity=None)}
    )
