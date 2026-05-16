"""
MCP server tool tests — call tool functions directly (no subprocess).
Phase 2 done criteria: 3+ tests per server, all pass.

Run: pytest tests/test_mcp_servers.py -v
"""

import pytest

# Import tool functions directly — faster than spawning MCP subprocess
from mcp_servers.clinical_trials_server import search_trials, get_trial_outcomes
from mcp_servers.pubmed_server import search_abstracts, get_full_abstract


# ── ClinicalTrials tests ──────────────────────────────────────────────────────

class TestSearchTrials:
    def test_returns_results(self):
        results = search_trials("NSCLC immunotherapy", max_results=5)
        assert len(results) > 0

    def test_result_schema(self):
        results = search_trials("breast cancer PARP inhibitor", max_results=3)
        assert len(results) > 0
        r = results[0]
        assert "nct_id" in r
        assert "title" in r
        assert "phase" in r
        assert "status" in r
        assert "conditions" in r
        assert "interventions" in r
        assert r["nct_id"].startswith("NCT")

    def test_phase_filter(self):
        results = search_trials("lung cancer", phase="3", max_results=10)
        assert len(results) > 0
        # All returned trials should be Phase 3 (or multi-phase including 3)
        for r in results:
            if r["phase"]:
                assert "3" in r["phase"] or r["phase"] == "PHASE3"

    def test_condition_filter(self):
        results = search_trials("pembrolizumab", condition="melanoma", max_results=5)
        assert len(results) > 0
        # At least one result should mention melanoma in conditions
        conditions_flat = [c.lower() for r in results for c in r["conditions"]]
        assert any("melanoma" in c for c in conditions_flat)

    def test_empty_query_graceful(self):
        # Should not raise — may return empty or generic results
        results = search_trials("zzz_no_match_xyzabc123", max_results=5)
        assert isinstance(results, list)


class TestGetTrialOutcomes:
    def test_known_trial(self):
        # NCT04507841 is in our DB (olaparib breast cancer trial)
        result = get_trial_outcomes("NCT04507841")
        assert result["nct_id"] == "NCT04507841"
        assert "title" in result
        assert "primary_outcomes" in result
        assert "secondary_outcomes" in result

    def test_outcomes_schema(self):
        result = get_trial_outcomes("NCT04507841")
        if result["primary_outcomes"]:
            o = result["primary_outcomes"][0]
            assert "measure" in o
            assert "description" in o
            assert "timeFrame" in o

    def test_returns_outcomes_populated(self):
        result = get_trial_outcomes("NCT04507841")
        # This trial should have at least one primary outcome
        assert len(result["primary_outcomes"]) > 0


# ── PubMed tests ─────────────────────────────────────────────────────────────

class TestSearchAbstracts:
    def test_returns_results(self):
        results = search_abstracts("pembrolizumab NSCLC overall survival", max_results=3)
        assert len(results) > 0

    def test_result_schema(self):
        results = search_abstracts("PARP inhibitor breast cancer BRCA", max_results=3)
        assert len(results) > 0
        r = results[0]
        assert "pmid" in r
        assert "title" in r
        assert "abstract_text" in r
        assert "authors" in r
        assert "journal" in r

    def test_abstract_truncated_in_list(self):
        results = search_abstracts("olaparib ovarian cancer", max_results=3)
        assert len(results) > 0
        # search_abstracts truncates to 400 chars
        for r in results:
            assert len(r["abstract_text"]) <= 400

    def test_date_filter(self):
        results = search_abstracts(
            "immunotherapy lung cancer", date_range_start="2022/01/01", max_results=5
        )
        assert isinstance(results, list)
        # Can't assert year without parsing pub_date, but should not raise


class TestGetFullAbstract:
    def test_known_pmid(self):
        # PMID 39366751 was fetched during ingestion (in our abstracts table)
        result = get_full_abstract("39366751")
        assert result["pmid"] == "39366751"
        assert len(result.get("abstract_text", "")) > 100

    def test_full_abstract_not_truncated(self):
        result = get_full_abstract("39366751")
        # Full abstract should be longer than the 400-char list truncation
        assert len(result.get("abstract_text", "")) > 400 or result.get("abstract_text") == ""

    def test_unknown_pmid_graceful(self):
        result = get_full_abstract("00000000")
        assert "pmid" in result
        assert "error" in result
