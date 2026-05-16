"""Unit tests for FAQ threshold statistics (no Pinecone / OpenAI)."""

import pytest

from scripts.analyze_faq_thresholds import percentile_nearest_rank, summarize_threshold_band


def test_percentile_nearest_rank_single_value():
    assert percentile_nearest_rank([0.5], 10) == 0.5
    assert percentile_nearest_rank([0.5], 90) == 0.5


def test_percentile_nearest_rank_median_of_ten():
    xs = [float(i) for i in range(1, 11)]  # 1..10
    # ceil(0.5 * 10) = 5 -> rank position 5 -> value 5.0
    assert percentile_nearest_rank(xs, 50) == 5.0


def test_percentile_empty():
    assert percentile_nearest_rank([], 50) is None


def test_summarize_threshold_band_clean_separation():
    match = [0.82, 0.79, 0.76, 0.71, 0.68, 0.65]
    escalate = [0.32, 0.41, 0.28]
    s = summarize_threshold_band(match, escalate, margin=0.02)
    assert s["escalate_max"] == 0.41
    assert s["floor_from_escalation"] == pytest.approx(0.43)
    assert "suggested_band_note" in s


def test_summarize_threshold_band_overlap_note():
    match = [0.55, 0.58, 0.60]
    escalate = [0.62, 0.70]
    s = summarize_threshold_band(match, escalate, margin=0.02)
    assert s["escalate_max"] == 0.70
    assert "ambiguous" in str(s["suggested_band_note"]).lower()
