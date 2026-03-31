"""
Standalone helper to fetch article titles from doi.org for a list of DOIs.

Reads a DOI CSV (same format as the downloader: required 'doi' column). Preserves
all other columns (e.g. 'name' for MOF refcode). Writes the same CSV with a 'title'
column filled by scraping https://doi.org/<doi>. Output is compatible with CatMOF
text mining: set paths.manuscript_data_csv to the output file. CatMOF expects
columns: name (MOF refcode), doi, title.

Run separately from the pipeline, e.g.:
  python -m catmof.text_mining_tools.title_fetcher --doi-csv dois.csv --output-csv manuscript_data_w_titles.csv

Requires optional deps: pip install catmof[text_mining_titles]
  (selenium, beautifulsoup4, webdriver-manager)
"""

import argparse
import os
import random
import sys
import time

import pandas as pd

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from bs4 import BeautifulSoup
except ImportError as e:
    raise ImportError(
        "Title fetcher requires selenium, beautifulsoup4, and webdriver-manager. "
        "Install with: pip install catmof[text_mining_titles]"
    ) from e

DOI_BASE_URL = "https://doi.org/"


def _get_title_for_doi(driver, doi, page_load_wait_sec=5):
    """Fetch doi.org page and return the <title> text, or '' on failure."""
    url = DOI_BASE_URL + doi
    try:
        driver.get(url)
        time.sleep(page_load_wait_sec)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        return (soup.title.string or "").strip() if soup.title else ""
    except Exception as e:
        print(f"Error processing DOI {doi}: {e}", file=sys.stderr)
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Fetch article titles from doi.org for DOIs in a CSV. Output is compatible with CatMOF manuscript_data_csv (columns: name, doi, title)."
    )
    parser.add_argument(
        "--doi-csv",
        required=True,
        help="CSV with a 'doi' column. Other columns (e.g. 'name') are preserved.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path (will include all input columns plus 'title').",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window (default is headless).",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=5.0,
        help="Seconds to wait after loading each doi.org page (default 5).",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=1.0,
        help="Min seconds between requests (default 1).",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=3.0,
        help="Max seconds between requests (default 3).",
    )
    args = parser.parse_args()

    headless = not args.no_headless

    if not os.path.isfile(args.doi_csv):
        print(f"DOI CSV not found: {args.doi_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.doi_csv)
    if "doi" not in df.columns:
        print("CSV must contain a 'doi' column.", file=sys.stderr)
        sys.exit(1)

    # Preserve all columns; add or overwrite 'title'
    if "title" not in df.columns:
        df["title"] = ""

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    else:
        options.add_argument("start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--disable-blink-features=AutomationControlled")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        for i, row in df.iterrows():
            doi = row["doi"]
            if pd.isna(doi) or str(doi).strip().lower() in ("", "unknown", "nan"):
                df.at[i, "title"] = ""
                continue
            doi = str(doi).strip()
            title = _get_title_for_doi(driver, doi, page_load_wait_sec=args.wait)
            df.at[i, "title"] = title
            if i < len(df) - 1:
                sleep_time = random.uniform(args.delay_min, args.delay_max)
                time.sleep(sleep_time)
    finally:
        driver.quit()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
