"""
NCBI Entrez (PubMed) API client.

Fetches abstracts linked to ClinicalTrials NCT IDs.
Uses Biopython's Bio.Entrez — no API key needed for low volume,
but ENTREZ_EMAIL must be set (NCBI policy).
"""

import os
import time
import logging
from typing import Optional

from Bio import Entrez
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

Entrez.email = os.getenv("ENTREZ_EMAIL", "medsignal@example.com")

BATCH_SIZE = 200
SLEEP_BETWEEN_BATCHES = 0.4   # stay well under NCBI's 3 req/s limit


class PubMedClient:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search(self, query: str, max_results: int = 20, date_start: Optional[str] = None) -> list[str]:
        """Return list of PMIDs matching query. date_start format: 'YYYY/MM/DD'."""
        params: dict = {"db": "pubmed", "term": query, "retmax": max_results, "usehistory": "y"}
        if date_start:
            params["mindate"] = date_start
            params["datetype"] = "pdat"

        with Entrez.esearch(**params) as handle:
            record = Entrez.read(handle)
        return list(record["IdList"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_abstracts(self, pmids: list[str]) -> list[dict]:
        """Fetch full abstract records for a list of PMIDs. Batched."""
        if not pmids:
            return []

        results = []
        for i in range(0, len(pmids), BATCH_SIZE):
            batch = pmids[i : i + BATCH_SIZE]
            id_str = ",".join(batch)
            with Entrez.efetch(db="pubmed", id=id_str, rettype="xml", retmode="xml") as handle:
                records = Entrez.read(handle)

            for article in records.get("PubmedArticle", []):
                parsed = self._parse_article(article)
                if parsed:
                    results.append(parsed)

            logger.info(f"Fetched batch {i // BATCH_SIZE + 1}: {len(batch)} abstracts")
            time.sleep(SLEEP_BETWEEN_BATCHES)

        return results

    def _parse_article(self, article: dict) -> Optional[dict]:
        try:
            medline = article["MedlineCitation"]
            pmid = str(medline["PMID"])
            article_data = medline["Article"]

            title = str(article_data.get("ArticleTitle", ""))

            abstract_texts = article_data.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_texts, list):
                abstract = " ".join(str(t) for t in abstract_texts)
            else:
                abstract = str(abstract_texts)

            authors = []
            for author in article_data.get("AuthorList", []):
                last = author.get("LastName", "")
                fore = author.get("ForeName", "")
                if last:
                    authors.append(f"{last} {fore}".strip())

            journal = str(article_data.get("Journal", {}).get("Title", ""))

            pub_date_struct = (
                article_data.get("Journal", {})
                .get("JournalIssue", {})
                .get("PubDate", {})
            )
            year = pub_date_struct.get("Year", "")
            month = pub_date_struct.get("Month", "01")
            pub_date = f"{year}-{month}-01" if year else None

            return {
                "pmid": pmid,
                "title": title,
                "abstract_text": abstract,
                "authors": authors,
                "journal": journal,
                "pub_date": pub_date,
            }
        except Exception as e:
            logger.warning(f"Failed to parse article: {e}")
            return None

    def fetch_for_nct_ids(self, nct_ids: list[str], max_per_nct: int = 3) -> list[dict]:
        """
        For each NCT ID, search PubMed for linked publications.
        Returns all abstracts found, with nct_id attached.
        """
        all_abstracts = []
        for nct_id in nct_ids:
            pmids = self.search(f"{nct_id}[si] OR {nct_id}[tw]", max_results=max_per_nct)
            if pmids:
                abstracts = self.fetch_abstracts(pmids)
                for a in abstracts:
                    a["nct_id"] = nct_id
                all_abstracts.extend(abstracts)
            time.sleep(SLEEP_BETWEEN_BATCHES)

        logger.info(f"Fetched {len(all_abstracts)} abstracts for {len(nct_ids)} NCT IDs")
        return all_abstracts
