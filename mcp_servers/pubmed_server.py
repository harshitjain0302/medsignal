"""
PubMed MCP Server.

Wraps the NCBI Entrez API as MCP tools.
Agents call these tools via MCP protocol — never Entrez directly.

Run standalone:  fastmcp dev mcp_servers/pubmed_server.py
Run in prod:     fastmcp run mcp_servers/pubmed_server.py

Tools:
  search_abstracts  — search PubMed by query → list of abstract summaries
  get_full_abstract — fetch full abstract + metadata for a PMID
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

from Bio import Entrez
from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent))

mcp = FastMCP("pubmed")

Entrez.email = os.getenv("ENTREZ_EMAIL", "medsignal@example.com")


@mcp.tool()
def search_abstracts(
    query: str,
    date_range_start: Optional[str] = None,
    max_results: int = 20,
) -> list[dict]:
    """
    Search PubMed abstracts via NCBI Entrez API.

    Args:
        query: Search query (e.g. 'pembrolizumab NSCLC overall survival')
        date_range_start: Filter by publication date — format 'YYYY/MM/DD'
        max_results: Max abstracts to return (default 20)

    Returns:
        List of dicts with pmid, title, abstract_text (truncated), authors, journal, pub_date
    """
    params: dict = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "usehistory": "y",
    }
    if date_range_start:
        params["mindate"] = date_range_start
        params["datetype"] = "pdat"

    with Entrez.esearch(**params) as handle:
        record = Entrez.read(handle)

    pmids = list(record["IdList"])
    if not pmids:
        return []

    time.sleep(0.35)  # NCBI rate limit: 3 req/s

    id_str = ",".join(pmids)
    with Entrez.efetch(db="pubmed", id=id_str, rettype="xml", retmode="xml") as handle:
        records = Entrez.read(handle)

    results = []
    for article in records.get("PubmedArticle", []):
        parsed = _parse_article(article)
        if parsed:
            # Truncate abstract for list view — full text via get_full_abstract
            parsed["abstract_text"] = parsed["abstract_text"][:400]
            results.append(parsed)

    return results


@mcp.tool()
def get_full_abstract(pmid: str) -> dict:
    """
    Fetch full abstract + metadata for a PubMed ID.

    Args:
        pmid: PubMed ID string (e.g. '39366751')

    Returns:
        Dict with pmid, title, full abstract_text, authors, journal, pub_date.
        Returns {pmid, error} if not found or NCBI returns an error.
    """
    try:
        with Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml") as handle:
            records = Entrez.read(handle)
    except Exception as e:
        return {"pmid": pmid, "error": str(e)}

    articles = records.get("PubmedArticle", [])
    if not articles:
        return {"pmid": pmid, "error": "not found"}

    return _parse_article(articles[0]) or {"pmid": pmid, "error": "parse failed"}


def _parse_article(article: dict) -> Optional[dict]:
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
    except Exception:
        return None


if __name__ == "__main__":
    mcp.run()
