"""
ClinicalTrials.gov MCP Server.

Wraps the ClinicalTrials.gov v2 REST API as MCP tools.
Agents call these tools via MCP protocol — never the API directly.

Run standalone:  fastmcp dev mcp_servers/clinical_trials_server.py
Run in prod:     fastmcp run mcp_servers/clinical_trials_server.py

Tools:
  search_trials      — search by query + filters → list of trial summaries
  get_trial_outcomes — fetch structured outcome measures for one trial
"""

import sys
from pathlib import Path
from typing import Optional

import httpx
from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.clinical_trials import _normalize_date  # reuse date normalizer

mcp = FastMCP("clinical-trials")

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def _get(params: dict) -> dict:
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(BASE_URL, params=params)
        resp.raise_for_status()
        return resp.json()


def _get_study(nct_id: str) -> dict:
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{BASE_URL}/{nct_id}", params={"format": "json"})
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
def search_trials(
    query: str,
    phase: Optional[str] = None,
    condition: Optional[str] = None,
    date_range_start: Optional[str] = None,
    date_range_end: Optional[str] = None,
    max_results: int = 20,
) -> list[dict]:
    """
    Search ClinicalTrials.gov for trials matching query and filters.

    Args:
        query: Free-text search (e.g. 'pembrolizumab NSCLC overall survival')
        phase: Phase filter — '2', '3', or '2 3' for both
        condition: Disease/condition (e.g. 'breast cancer')
        date_range_start: ISO date string 'YYYY-MM-DD' — filter by start date
        date_range_end: ISO date string 'YYYY-MM-DD'
        max_results: Max trials to return (default 20, max 100)

    Returns:
        List of trial dicts with nct_id, title, phase, status, conditions, interventions
    """
    params: dict = {
        "query.term": query,
        "pageSize": min(max_results, 100),
        "format": "json",
    }

    if condition:
        params["query.cond"] = condition
    if phase:
        params["aggFilters"] = f"phase:{phase}"
    if date_range_start:
        params["query.term"] = f"{query} AREA[StartDate]RANGE[{date_range_start}, MAX]"

    data = _get(params)
    studies = data.get("studies", [])

    results = []
    for s in studies:
        proto = s.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design_mod = proto.get("designModule", {})
        cond_mod = proto.get("conditionsModule", {})
        interv_mod = proto.get("armsInterventionsModule", {})

        phases = design_mod.get("phases", [])
        results.append({
            "nct_id": id_mod.get("nctId", ""),
            "title": id_mod.get("briefTitle", ""),
            "phase": phases[0] if phases else None,
            "status": status_mod.get("overallStatus", ""),
            "conditions": cond_mod.get("conditions", []),
            "interventions": [
                iv.get("name", "") for iv in interv_mod.get("interventions", [])
            ],
            "start_date": _normalize_date(
                status_mod.get("startDateStruct", {}).get("date")
            ),
            "completion_date": _normalize_date(
                status_mod.get("completionDateStruct", {}).get("date")
            ),
        })

    return results


@mcp.tool()
def get_trial_outcomes(nct_id: str) -> dict:
    """
    Get structured outcome measures for a specific trial.

    Args:
        nct_id: ClinicalTrials.gov NCT ID (e.g. 'NCT04507841')

    Returns:
        Dict with nct_id, title, primary_outcomes, secondary_outcomes.
        Each outcome: {measure, description, timeFrame}
    """
    data = _get_study(nct_id)
    proto = data.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    outcomes_mod = proto.get("outcomesModule", {})

    return {
        "nct_id": nct_id,
        "title": id_mod.get("briefTitle", ""),
        "primary_outcomes": [
            {
                "measure": o.get("measure", ""),
                "description": o.get("description", ""),
                "timeFrame": o.get("timeFrame", ""),
            }
            for o in outcomes_mod.get("primaryOutcomes", [])
        ],
        "secondary_outcomes": [
            {
                "measure": o.get("measure", ""),
                "description": o.get("description", ""),
                "timeFrame": o.get("timeFrame", ""),
            }
            for o in outcomes_mod.get("secondaryOutcomes", [])
        ],
    }


if __name__ == "__main__":
    mcp.run()
