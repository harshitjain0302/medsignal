"""
ClinicalTrials.gov v2 REST API client.

Fetches oncology trials (Phase 2/3) with rate-limit-safe pagination.
Raw JSON cached to disk before DB insertion — allows reruns without re-hitting the API.

API quirks discovered during development:
  - `fields` param causes 400 — v2 API returns full study objects; no field selection supported
  - `filter.phase` param does not exist in v2 — use `aggFilters=phase:<values>`
  - `aggFilters` is a single-value param (passing it twice = 400)
  - Multiple values within one filter use SPACE separator: `aggFilters=phase:2 3`
    (comma = filter-pair separator, pipe = returns 0 results, two params = 400)
  - Space must be passed as %20 when building URLs manually; httpx params dict handles this
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

CACHE_DIR = Path("data/raw/clinical_trials")


def _normalize_date(date_str: Optional[str]) -> Optional[str]:
    """
    ClinicalTrials.gov v2 returns partial dates (YYYY-MM or YYYY-MM-DD or YYYY).
    PostgreSQL DATE type requires YYYY-MM-DD. Pad missing parts with -01.
    """
    if not date_str:
        return None
    parts = date_str.split("-")
    if len(parts) == 1:       # "2009" → "2009-01-01"
        return f"{parts[0]}-01-01"
    if len(parts) == 2:       # "2009-03" → "2009-03-01"
        return f"{parts[0]}-{parts[1]}-01"
    return date_str           # already "YYYY-MM-DD"


class ClinicalTrialsClient:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get_page(self, params: dict) -> dict:
        """Single paginated request with retry on transient errors."""
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(BASE_URL, params=params)
            resp.raise_for_status()
            return resp.json()

    def fetch_oncology_trials(
        self,
        max_trials: int = 10000,
        phases: tuple[str, ...] = ("2", "3"),
        condition: str = "cancer",
        page_size: int = 100,
        sleep_between_pages: float = 0.5,
    ) -> list[dict]:
        """
        Fetch oncology trials from ClinicalTrials.gov v2 API.

        Uses condition=cancer + aggFilters=phase:<n> filter. Returns raw study dicts.
        Caches each page to disk so reruns skip already-fetched pages.
        sleep_between_pages: respects API rate limits (see pitfalls in master prompt).

        phases: phase numbers as strings e.g. ("2", "3") — joined with space for aggFilters.
          aggFilters single-value param; space = OR between values within one filter.
        """
        all_studies: list[dict] = []
        next_page_token: Optional[str] = None
        page_num = 0

        # Space-separated phase values: "2 3" → aggFilters=phase:2%203
        phase_filter = f"phase:{' '.join(phases)}"

        # Sentinel file marks a complete fetch run. pageTokens expire within hours,
        # so pagination must complete in one session. If sentinel missing, wipe cache
        # and re-fetch from scratch.
        sentinel = self.cache_dir / "COMPLETE"
        if not sentinel.exists():
            existing = list(self.cache_dir.glob("page_*.json"))
            if existing:
                logger.info(f"Incomplete cache detected ({len(existing)} pages). Wiping and re-fetching.")
                for f in existing:
                    f.unlink()

        if sentinel.exists():
            # Load all cached pages in order
            logger.info("Complete cache found — loading from disk.")
            page_files = sorted(self.cache_dir.glob("page_*.json"))
            for pf in page_files:
                data = json.loads(pf.read_text())
                all_studies.extend(data.get("studies", []))
                if len(all_studies) >= max_trials:
                    break
            logger.info(f"Loaded {len(all_studies)} trials from cache.")
            return all_studies[:max_trials]

        # Live fetch — token chain must be maintained in-memory for the full run
        while len(all_studies) < max_trials:
            cache_file = self.cache_dir / f"page_{page_num:04d}.json"

            # pageToken encodes full query state — do NOT re-pass query/filter params
            # alongside it or the API returns 400.
            if next_page_token:
                params: dict = {
                    "pageToken": next_page_token,
                    "pageSize": min(page_size, max_trials - len(all_studies)),
                    "format": "json",
                }
            else:
                params = {
                    "query.cond": condition,
                    "aggFilters": phase_filter,
                    "pageSize": min(page_size, max_trials - len(all_studies)),
                    "format": "json",
                }

            logger.info(f"Fetching page {page_num} (total so far: {len(all_studies)})")
            data = self._get_page(params)
            cache_file.write_text(json.dumps(data))
            time.sleep(sleep_between_pages)

            studies = data.get("studies", [])
            if not studies:
                logger.info("No more studies returned — pagination complete.")
                break

            all_studies.extend(studies)
            page_num += 1
            next_page_token = data.get("nextPageToken")

            if not next_page_token:
                logger.info("No nextPageToken — reached last page.")
                break

        sentinel.write_text("ok")
        logger.info(f"Fetched {len(all_studies)} trials total. Cache marked complete.")
        return all_studies[:max_trials]

    def parse_trial(self, study: dict) -> dict:
        """
        Normalize a raw ClinicalTrials.gov v2 study dict into our DB schema shape.
        Returns a flat dict matching the `trials` table columns.
        """
        proto = study.get("protocolSection", {})
        id_module = proto.get("identificationModule", {})
        status_module = proto.get("statusModule", {})
        desc_module = proto.get("descriptionModule", {})
        design_module = proto.get("designModule", {})
        conditions_module = proto.get("conditionsModule", {})
        interventions_module = proto.get("armsInterventionsModule", {})
        outcomes_module = proto.get("outcomesModule", {})

        nct_id = id_module.get("nctId", "")
        phases = design_module.get("phases", [])
        phase = phases[0] if phases else None

        conditions = conditions_module.get("conditions", [])
        interventions = [
            iv.get("name", "") for iv in interventions_module.get("interventions", [])
        ]

        primary_outcomes = [
            {
                "measure": o.get("measure", ""),
                "description": o.get("description", ""),
                "timeFrame": o.get("timeFrame", ""),
            }
            for o in outcomes_module.get("primaryOutcomes", [])
        ]
        secondary_outcomes = [
            {
                "measure": o.get("measure", ""),
                "description": o.get("description", ""),
                "timeFrame": o.get("timeFrame", ""),
            }
            for o in outcomes_module.get("secondaryOutcomes", [])
        ]

        # Full text for chunking — brief summary + detailed description + eligibility
        eligibility_module = proto.get("eligibilityModule", {})
        full_text_parts = [
            id_module.get("briefTitle", ""),
            desc_module.get("briefSummary", ""),
            desc_module.get("detailedDescription", ""),
            eligibility_module.get("eligibilityCriteria", ""),
        ]
        full_text = "\n\n".join(p for p in full_text_parts if p)

        start_date = _normalize_date(status_module.get("startDateStruct", {}).get("date"))
        completion_date = _normalize_date(status_module.get("completionDateStruct", {}).get("date"))

        return {
            "nct_id": nct_id,
            "title": id_module.get("briefTitle", ""),
            "phase": phase,
            "conditions": conditions,
            "interventions": interventions,
            "status": status_module.get("overallStatus", ""),
            "start_date": start_date,
            "completion_date": completion_date,
            "primary_outcomes": primary_outcomes,
            "secondary_outcomes": secondary_outcomes,
            "full_text": full_text,
            "raw_json": study,
        }

    def fetch_and_parse(self, max_trials: int = 10000) -> list[dict]:
        """Convenience method: fetch raw + parse. Returns list of parsed trial dicts."""
        raw = self.fetch_oncology_trials(max_trials=max_trials)
        return [self.parse_trial(s) for s in raw]
