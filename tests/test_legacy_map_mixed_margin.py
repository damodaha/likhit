"""Tests for VOL-218's mixed letter+digit margin gate on the legacy map chooser.

**What the gate is for.** The six legacy maps fall into two families that swap the
number rows: ``Preeti``/``Kantipur``/``Sagarmatha``/``Spins`` put the Devanagari
digits on the SHIFTED row ``!@#$%^&*()`` and read ``0123456789`` as consonants, while
``PCS NEPALI``/``FONTASY_HIMALI_TT`` do the reverse. Choosing the wrong family
therefore does not *garble* a span, it TRANSPOSES letters and digits -- and on a
document that types money on one row and place names on the other, a single wrong
choice produces both directions at once.

That is what happened to ``3719__...Humla Sarkegad`` on v13 -> v14. Its ``Felix
Titling`` face flipped ``PCS NEPALI`` -> ``Preeti``, and in one table 49
unshifted-row keystrokes became consonants (``?= 10875.00`` -> ``ru. ...``, the money
column) while 23 shifted-row keystrokes became digits (``uf]&L`` -> ``go-7-i`` for
``gothi``, a place name). Measured in ``oag-corpus/runs/vol218/`` and recorded in
``FINDING-16``/``FINDING-19``.

**Why nothing else catches it.** A wrong-family reading scores ``penalty 0`` and a
high ``ratio``, because every keystroke still lands on Devanagari. On this face all
six maps tie on ``hits``, ``penalty``, ``stranded`` AND ``attested``, so the decision
fell through to ``ratio``, where ``Preeti`` won by **0.000391** -- inside the band the
ranking's own docstring calls unusable. The document-level attested screen saw only
``net_attested_delta -2``, because seven gained ``ru`` nearly cancelled the loss.

**Why a MARGIN and not just another axis.** The bare term was priced corpus-wide
first: below ``attested`` it costs nothing but leaves ``4834...kharpunath`` damaged;
above ``attested`` it repairs all four damaged documents but takes ``attested -5`` and
makes four flips that are not repairs, because it speaks on spans where its advantage
is a single token. Gated on a margin it makes 6 flips at M=5 -- a strict subset of the
ungated 11 -- keeps all four repairs, and costs ``attested -2``, all of which is one
document's own repair.

**The gate is OPT-IN and OFF by default**, because which margin ships is an open board
decision. The first test below is the one that matters most: with the gate off the
chooser must be indistinguishable from the shipped one.
"""

import pytest

from likhit.errors import ExtractionError
from likhit.extractors import font_based as fb

# Distinct per axis, so no assertion in this file can pass because two axes happen to
# hold the same number. Run `9c7a9a3b`'s first attempt at this measurement used a span
# that scored 0.0 on every axis and therefore reported a missing axis as PRESENT.
_SENTINELS = {
    "hits": 11.0,
    "penalty": 13.0,
    "stranded": 0.0,  # 0 so the forgiveness clamp cannot alias another axis
    "figures": 19.0,
    "attested": 23.0,
    "ratio": 29.0,
    "devanagari": 31.0,
    "mixed": 0.0,
}


def _derive_eligible_index() -> int:
    """Where the indicator sits, DERIVED from the two keys rather than hardcoded.

    Adding an axis above the indicator shifts its position. A hardcoded index would
    then silently move every positional assertion in this file onto the wrong axis --
    which is exactly the class of failure `9c7a9a3b` measured in the production code.
    """

    shipped = fb._map_ranking_key(_SENTINELS)
    gated = fb._map_ranking_key_margin_gated(threshold=-1.0)(_SENTINELS)
    assert len(gated) == len(shipped) + 1, (shipped, gated)
    positions = [
        index
        for index in range(len(gated))
        if gated[:index] + gated[index + 1 :] == shipped
    ]
    assert len(positions) == 1, f"indicator position is not unique: {positions}"
    return positions[0]


_ELIGIBLE_INDEX = _derive_eligible_index()


# The real ``Felix Titling`` aggregate from
# ``3719__1613986243Humla Sarkegad Gaupalika207475.pdf`` -- 30 spans on page 21,
# concatenated with no separator, which is how ``detect_content_legacy_fonts`` builds
# the decision unit. Legacy keystrokes are ASCII, so this is the literal byte content
# of the PDF's text; the one non-ASCII character is escaped.
AGGREGATE_3719 = (
    ";fd'bflos eag of]hgf pkef]Qmf ;ldltsf] lan e'QmfgL ubf{ s/ s\u00a7f ug'{ kg]{"
    " /sd ?= 10875.00 s\u00a7f gePsf] c;'n ul/ bflvnf ug]{' kg]{   uf]&L b]lv /f]l"
    "*sf]^ ;Dd #f]*]^f] af^f] of]hgf pkef]Qmf ;ldltsf] lan e'QmfgL ubf{ s/ sf"
    " ug'{ kg]{ /sd ?= 19305.00 sf gePsf] /sd c;'n ul/ bflvnf ug]{' kg a/fO{ "
    "b]lv /ftf *f*f ;Dd #f]*]^f] af^f] of]hgf pkef]Qmf ;ldltsf] lan e'QmfgL u"
    "bf{ s/ sf ug'{ kg]{ /sd ?= 14888.00 sf gePsf] /sd c;'n ul/ bflvnf ug]{' "
    "kg enfbL b]lv uf]&L ;Dd u|fld)f ;*s of]hgf pkef]Qmf ;ldltsf] lan e'QmfgL"
    " ubf{ s/ sf ug'{ kg]{ /sd ?= 10858.00 sf gePsf] /sd c;'n ul/ bflvnf ug]{"
    "' kg]{ ;s]{uf( b]lv *f*f;fof;Dd u|fld)f ;*s of]hgf pkef]Qmf ;ldltsf] lan"
    " e'QmfgL ubf{ s/ s\u00a7f ug'{ kg]{ /sd ?= 12921.00 s\u00a7f gePsf] /sd c;'n ul/ b"
    "flvnf ug]{' kg]{ l/kuf( hlaw't afw pkef]Qmf ;ldltsf] lan e'QmfgL ubf{ s/"
    " sf ug'{ kg]{ /sd ?= 15318.00 sf gePsf] /sd c;'n ul/ bflvnf ug]{' kg]{ c"
    "+z'adf{ cfwf/e't laWofno eag lddf{)f of]hgf pkef]Qmf ;ldltsf] lan e'Qmfg"
    "L ubf{ s/ sf ug'{ kg]{ /sd ?= 13786.00 sf gePsf] /sd c;'n ul/ bflvnf ug]"
    "{' kg]{ "
)


def test_aggregate_fixture_is_the_measured_unit():
    """Guard the fixture itself: 1,016 characters over 30 spans (FINDING 16)."""

    assert len(AGGREGATE_3719) == 1016


class TestMixedLetterDigitCount:
    """The measure. It is comparative between readings of ONE span, never absolute."""

    def test_counts_the_damage_forms(self):
        # go-7-i for gothi, and graami-0-aa for graamin: the transposed readings.
        assert fb._mixed_letter_digit_count("\u0917\u094b\u096d\u0940") == 1
        assert (
            fb._mixed_letter_digit_count(
                "\u0917\u094d\u0930\u093e\u092e\u093f\u0966\u093e"
            )
            == 1
        )

    def test_does_not_count_the_correct_readings(self):
        # gothi and graamin as they should read: letters only.
        assert fb._mixed_letter_digit_count("\u0917\u094b\u0920\u0940") == 0
        assert (
            fb._mixed_letter_digit_count("\u0917\u094d\u0930\u093e\u092e\u093f\u0923")
            == 0
        )

    def test_the_letter_class_is_CONSONANTS_ONLY_so_an_ordinal_scores_zero(self):
        """``10au.`` -- a legitimate Nepali ordinal -- does NOT enter this count.

        Pinned because it is surprising and the natural assumption is the opposite.
        The letter class is U+0915-U+0939 plus the nukta consonants U+0958-U+095F:
        **consonants only**. The independent vowel ``au`` is U+0914 and the anusvara
        is U+0902, and neither is in the letter class OR the mark class
        (U+093A-U+094F, U+0951-U+0957, U+0962-U+0963), so they break the token and
        ``10`` is left as digits with no letter beside them.

        These classes are character-for-character the ones the corpus-wide arm sweep
        used (``runs/vol218/sweep_margin_gate_corpus_0aa6842c.py``). That is the
        requirement, not an accident: the M=5 result this gate ships was measured with
        exactly this measure, so widening the class here would silently invalidate
        every flip, repair and ``attested`` figure on the issue.

        Note this cuts the safe way -- a narrower letter class can only *miss* mixed
        tokens, and the term is a comparative one, so a form both candidates produce
        cancels regardless.
        """

        assert fb._mixed_letter_digit_count("\u0967\u0966\u0914\u0902") == 0
        # A consonant beside a digit is what the class is for, and it does count.
        assert fb._mixed_letter_digit_count("\u0967\u0966\u0915") == 1

    def test_ignores_ascii_digits(self):
        """ASCII 7 beside a Devanagari letter is undecoded keystroke, not this class."""

        assert fb._mixed_letter_digit_count("\u0917\u094b7\u0940") == 0

    def test_character_classes_are_asserted_not_trusted(self):
        """A decomposed class literal compiles and silently reclassifies marks."""

        fb._assert_mixed_classes_hold()


class TestMarginSetting:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv(fb._MIXED_MARGIN_ENV_VAR, raising=False)
        assert fb._mixed_margin_setting() is None

    def test_blank_is_off(self, monkeypatch):
        monkeypatch.setenv(fb._MIXED_MARGIN_ENV_VAR, "   ")
        assert fb._mixed_margin_setting() is None

    def test_parses_an_integer(self, monkeypatch):
        monkeypatch.setenv(fb._MIXED_MARGIN_ENV_VAR, "5")
        assert fb._mixed_margin_setting() == 5

    def test_refuses_garbage_rather_than_disabling_itself(self, monkeypatch):
        """A gate that quietly turns itself off makes a build unfalsifiable."""

        monkeypatch.setenv(fb._MIXED_MARGIN_ENV_VAR, "five")
        with pytest.raises(ExtractionError):
            fb._mixed_margin_setting()

    def test_refuses_zero(self, monkeypatch):
        """M=0 would mean 'any advantage at all', which is the UNGATED arm."""

        monkeypatch.setenv(fb._MIXED_MARGIN_ENV_VAR, "0")
        with pytest.raises(ExtractionError):
            fb._mixed_margin_setting()


class TestRankingKeyIsAnIndicator:
    """The inserted term promotes the eligible SET; it does not order within it."""

    @staticmethod
    def _validity(mixed, attested, figures=0.0):
        return {
            "hits": 3.0,
            "penalty": 0.0,
            "stranded": 0.0,
            # `figures` is VOL-289's axis (`46fd302`). It is in this fixture because
            # the shipped key reads it; a fixture that omits an axis the key reads is
            # how these tests came to pin the tuple's OLD positions.
            "figures": float(figures),
            "attested": float(attested),
            "ratio": 0.99,
            "devanagari": 100.0,
            "mixed": float(mixed),
        }

    def test_eligibility_is_a_step_not_a_gradient(self):
        key = fb._map_ranking_key_margin_gated(threshold=8.0)
        # Both are eligible at the threshold, so the term cannot separate them and
        # `attested` decides -- 2 beats 5 on mixed, but loses on attested.
        low_mixed = key(self._validity(mixed=2, attested=10))
        at_threshold = key(self._validity(mixed=8, attested=20))
        assert at_threshold > low_mixed

    def test_an_ineligible_candidate_is_demoted_below_attested(self):
        key = fb._map_ranking_key_margin_gated(threshold=8.0)
        eligible_weak = key(self._validity(mixed=8, attested=1))
        ineligible_strong = key(self._validity(mixed=9, attested=99))
        assert eligible_weak > ineligible_strong

    def test_a_negative_threshold_makes_the_term_constant(self):
        """The silent case: no candidate can be eligible, so shipped order holds."""

        key = fb._map_ranking_key_margin_gated(threshold=-3.0)
        assert key(self._validity(mixed=0, attested=5))[_ELIGIBLE_INDEX] == 0.0
        assert key(self._validity(mixed=99, attested=5))[_ELIGIBLE_INDEX] == 0.0

    def test_the_term_sits_below_figures_and_above_attested(self):
        """`both_fig_first` (card `a5f18dcb`): `figures` outranks the indicator.

        Positions are asserted with DISTINCT values per axis, so an axis that moves
        cannot be masked by another axis holding the same number -- the failure mode
        that made run `9c7a9a3b`'s first introspection pass report a clean tree.
        """

        key = fb._map_ranking_key_margin_gated(threshold=0.0)
        got = key(self._validity(mixed=0, attested=7, figures=19))
        assert got[2] == 0.0  # -max(stranded - forgiveness, 0)
        assert got[3] == 19.0  # figures, VOL-289's axis, ABOVE the indicator
        assert got[_ELIGIBLE_INDEX] == 1.0  # the eligibility indicator
        assert got[5] == 7.0  # attested, below both


class TestGateOffIsShipped:
    """The default must be indistinguishable from the chooser without this change.

    ⚠️ These three asserted the LITERAL ``"Preeti"`` until run `9c7a9a3b`. That was
    `ecf857c`'s answer for 3719, and `46fd302` -- VOL-289's ``figures`` axis -- makes it
    ``PCS NEPALI``, so all three failed on the joint tip for a reason that has nothing
    to do with the gate. **A test that pins a literal decision pins the BASE, not the
    behaviour it means to protect.** The property here is a comparison: off must equal
    the chooser with this change structurally absent, whatever that chooser decides.
    """

    @staticmethod
    def _shipped_choice():
        """The chooser without this change: the shipped key, no mixed term at all."""

        return fb._choose_legacy_map_ranked(
            AGGREGATE_3719, fb._map_ranking_key, mixed_threshold=None
        )

    def test_env_unset_leaves_3719_on_the_shipped_map(self, monkeypatch):
        monkeypatch.delenv(fb._MIXED_MARGIN_ENV_VAR, raising=False)
        choice = fb.choose_legacy_map_detailed(AGGREGATE_3719)
        assert choice.map_key == self._shipped_choice().map_key

    def test_explicit_none_leaves_3719_on_the_shipped_map(self):
        choice = fb.choose_legacy_map_detailed(AGGREGATE_3719, mixed_margin=None)
        assert choice.map_key == self._shipped_choice().map_key

    def test_an_explicit_none_overrides_a_set_environment(self, monkeypatch):
        """A caller that means OFF must not be overridden by an env var."""

        monkeypatch.setenv(fb._MIXED_MARGIN_ENV_VAR, "5")
        choice = fb.choose_legacy_map_detailed(AGGREGATE_3719, mixed_margin=None)
        assert choice.map_key == self._shipped_choice().map_key

    def test_off_computes_no_mixed_term_at_all(self, monkeypatch):
        monkeypatch.delenv(fb._MIXED_MARGIN_ENV_VAR, raising=False)
        choice = fb.choose_legacy_map_detailed(AGGREGATE_3719)
        assert "mixed" not in (choice.validity or {})


class TestGateOnRepairs3719:
    """3719 must read ``PCS NEPALI`` when the gate is on -- however it gets there.

    ⚠️ **On a tree carrying VOL-289's ``figures`` axis these pass without the gate
    doing any of the work.** Pass 1 already decides ``PCS NEPALI``, so the threshold is
    `0 - M` and the gate is silent. That is not a weaker result -- the document is
    repaired either way -- but it means these tests do not, on such a tree, demonstrate
    the gate. `TestTheGatedKeyCannotDriftFromTheShipped` is what protects them there.

    Three of these asserted the intermediate quantities `13`, `8.0` and the literal
    ``"Preeti"`` until run `9c7a9a3b`; each of those is a property of `ecf857c`'s
    pass-1 winner, not of the gate, and all three broke on the joint tip. They are now
    written against quantities the tree reports.
    """

    @staticmethod
    def _mixed_of(choice):
        return fb._mixed_letter_digit_count(
            fb.get_converter_for_map(choice.map_key)(AGGREGATE_3719)
        )

    def test_margin_five_restores_pcs_nepali(self):
        choice = fb.choose_legacy_map_detailed(AGGREGATE_3719, mixed_margin=5)
        assert choice.map_key == "PCS NEPALI"

    def test_the_repair_removes_every_mixed_token(self):
        shipped = fb.choose_legacy_map_detailed(AGGREGATE_3719, mixed_margin=None)
        gated = fb.choose_legacy_map_detailed(AGGREGATE_3719, mixed_margin=5)
        shipped_mixed = self._mixed_of(shipped)
        gated_mixed = self._mixed_of(gated)
        # The invariant is directional, and it is the whole point of the issue: the
        # gated reading carries NO transposed letter/digit tokens, and can never carry
        # more than the shipped one.
        assert gated_mixed == 0
        assert gated_mixed <= shipped_mixed

    def test_the_environment_variable_is_an_equivalent_route(self, monkeypatch):
        monkeypatch.setenv(fb._MIXED_MARGIN_ENV_VAR, "5")
        assert fb.choose_legacy_map_detailed(AGGREGATE_3719).map_key == "PCS NEPALI"

    def test_the_repaired_reading_spells_the_place_names(self):
        """The point of the exercise: real words, not digits."""

        gated = fb.choose_legacy_map_detailed(AGGREGATE_3719, mixed_margin=5)
        reading = fb.decode_with_legacy_map(AGGREGATE_3719, gated)
        assert "\u0917\u094b\u0920\u0940" in reading  # gothi
        assert "\u0917\u094b\u096d\u0940" not in reading  # go-7-i

    def test_a_margin_wider_than_the_advantage_stays_silent(self):
        """A margin no advantage can clear must leave the shipped decision alone.

        Stated as a comparison rather than as the literal ``"Preeti"``: which map that
        is depends on the tree's other axes, and the property does not.
        """

        shipped = fb.choose_legacy_map_detailed(AGGREGATE_3719, mixed_margin=None)
        assert (
            fb.choose_legacy_map_detailed(AGGREGATE_3719, mixed_margin=99).map_key
            == shipped.map_key
        )


class TestGateCannotManufactureADecision:
    """Where the shipped chooser abstains the gate must stay silent.

    This is the property that bounds the arm's blast radius: it can move a span from
    one map to another, but it can never bring a span the accept gate rejected into
    the transcript. Asserted on a span that really does abstain, not on the code.
    """

    ABSTAINING = "\u0917 \u0916"  # two Devanagari letters: no legacy keystrokes at all

    def test_shipped_abstains_here(self):
        assert (
            fb.choose_legacy_map_detailed(self.ABSTAINING, mixed_margin=None).map_key
            is None
        )

    def test_the_gate_also_abstains_here(self):
        assert (
            fb.choose_legacy_map_detailed(self.ABSTAINING, mixed_margin=5).map_key
            is None
        )

    def test_an_empty_span_abstains_both_ways(self):
        assert fb.choose_legacy_map_detailed("", mixed_margin=None).map_key is None
        assert fb.choose_legacy_map_detailed("", mixed_margin=5).map_key is None

    @staticmethod
    def _count_passes(monkeypatch):
        """Record the ``mixed_threshold`` of every ranking pass, in order."""

        passes: list = []
        original = fb._choose_legacy_map_ranked

        def counting(text, ranking_key, mixed_threshold):
            passes.append(mixed_threshold)
            return original(text, ranking_key, mixed_threshold)

        monkeypatch.setattr(fb, "_choose_legacy_map_ranked", counting)
        return passes

    def test_never_asks_for_a_converter_for_a_non_decision(self, monkeypatch):
        """The guard is asserted directly, because its absence is otherwise INVISIBLE.

        Dropping ``shipped.map_key is None`` from the guard changes neither the result
        nor the number of ranking passes: the threshold lookup raises on a ``None`` map
        key and the surrounding ``except`` returns the shipped choice, *before* pass 2
        is reached. A mutation run confirmed it -- the mutant survived both an
        outcome-only test and a pass-counting one, because it is a semantically
        equivalent mutant.

        What does separate the two is whether the code ASKS for a converter it has no
        business asking for. Enforcing an invariant by relying on a downstream
        exception is the shape that breaks silently the day that lookup is made
        tolerant of ``None``, so the invariant is pinned here instead:
        a span with no decision never reaches the converter lookup.
        """

        seen: list = []
        original = fb.get_converter_for_map

        def recording(map_key):
            seen.append(map_key)
            return original(map_key)

        monkeypatch.setattr(fb, "get_converter_for_map", recording)
        assert (
            fb.choose_legacy_map_detailed(self.ABSTAINING, mixed_margin=5).map_key
            is None
        )
        assert None not in seen, "asked for a converter for a None map key"

    def test_no_second_pass_at_all_when_the_shipped_chooser_abstains(self, monkeypatch):
        """Outcome-level companion to the above: exactly one ranking pass runs."""

        passes = self._count_passes(monkeypatch)
        assert (
            fb.choose_legacy_map_detailed(self.ABSTAINING, mixed_margin=5).map_key
            is None
        )
        assert passes == [None], f"expected one shipped pass only, got {passes}"

    def test_two_passes_when_the_shipped_chooser_decides(self, monkeypatch):
        """Positive control for the counter above: on 3719 the gate really does re-rank."""

        shipped = fb.choose_legacy_map_detailed(AGGREGATE_3719, mixed_margin=None)
        expected_threshold = float(
            fb._mixed_letter_digit_count(
                fb.get_converter_for_map(shipped.map_key)(AGGREGATE_3719)
            )
            - 5
        )
        passes = self._count_passes(monkeypatch)
        assert (
            fb.choose_legacy_map_detailed(AGGREGATE_3719, mixed_margin=5).map_key
            == "PCS NEPALI"
        )
        # Pass 1 shipped (None), then pass 2 at mixed(pass-1 winner) - margin. The
        # threshold is DERIVED, not the literal 8.0: that number was `13 - 5` on
        # `ecf857c`, and `46fd302` makes the pass-1 winner's count 0.
        assert passes == [None, expected_threshold], (
            f"expected [None, {expected_threshold}], got {passes}"
        )


class TestTheGatedKeyCannotDriftFromTheShipped:
    """The regression run `9c7a9a3b` measured, pinned as a RELATIONSHIP between the
    two keys rather than as either key's contents -- so it keeps biting as axes are
    added.

    History, because it is the reason these tests exist. The gate's pass-2 key was
    written as a hand copy of the shipped tuple when that tuple had six elements.
    `46fd302` then inserted VOL-289's ``figures`` axis at index 3, and the copy had
    ``ELIGIBLE`` at that index -- so pass 2 OVERWROTE the figures slot, and because
    :func:`choose_legacy_map_detailed` returns pass 2 on every deciding unit, enabling
    the gate deleted the figures axis corpus-wide. Cherry-picking the two commits
    together auto-merges CLEAN, so nothing warned. Measured footprint on the joint
    tip: all six figures repairs reverted, including `3719`, the document the gate
    exists to repair.
    """

    @staticmethod
    def _validity(**overrides):
        return {**_SENTINELS, **overrides}

    def test_the_gated_key_is_the_shipped_key_plus_exactly_one_element(self):
        """Delete the indicator and what is left must be the shipped tuple EXACTLY.

        This is the drift detector: it fails the moment an axis exists in the shipped
        key and not in the gated one, whatever that axis is and wherever it sits.
        """

        for mixed, threshold in ((0.0, 5.0), (9.0, -1.0), (3.0, 3.0), (7.0, 0.0)):
            validity = self._validity(mixed=mixed)
            shipped = fb._map_ranking_key(validity)
            gated = fb._map_ranking_key_margin_gated(threshold=threshold)(validity)
            stripped = gated[:_ELIGIBLE_INDEX] + gated[_ELIGIBLE_INDEX + 1 :]
            assert stripped == shipped, f"mixed={mixed} threshold={threshold}"

    def test_every_shipped_axis_survives_into_the_gated_key(self):
        """Axis by axis, by VALUE, so a vanished axis cannot hide behind a zero."""

        gated = fb._map_ranking_key_margin_gated(threshold=-1.0)(dict(_SENTINELS))
        for axis, value in _SENTINELS.items():
            if axis in {"mixed", "stranded", "penalty"}:
                continue  # `mixed` is not an axis; `stranded`/`penalty` enter signed
            assert value in gated, f"{axis} ({value}) is missing from the gated key"

    def test_the_silent_case_leaves_the_shipped_decision_alone_end_to_end(self):
        """The invariant the docstring claimed and the joint tip violated.

        On a tree carrying ``figures``, `3719`'s span is already repaired by that axis,
        so pass 1 wins with mixed 0 and the threshold `0 - 5` makes every candidate
        ineligible -- the silent case. The gate must then return pass 1's decision. On
        the pre-fix joint tip it returned ``Preeti``, re-damaging the document.
        """

        shipped = fb.choose_legacy_map_detailed(AGGREGATE_3719, mixed_margin=None)
        silent = fb.choose_legacy_map_detailed(AGGREGATE_3719, mixed_margin=5)
        assert silent.map_key == shipped.map_key

    def test_a_constant_indicator_cannot_reorder_candidates(self):
        """Why the silent case holds by construction and not by inspection."""

        key = fb._map_ranking_key_margin_gated(threshold=-1.0)
        weak = self._validity(figures=1.0, attested=1.0)
        strong = self._validity(figures=9.0, attested=1.0)
        assert key(weak)[_ELIGIBLE_INDEX] == key(strong)[_ELIGIBLE_INDEX] == 0.0
        assert (key(strong) > key(weak)) == (
            fb._map_ranking_key(strong) > fb._map_ranking_key(weak)
        )
