"""
Standalone helper to download article HTML/XML files by DOI.

Uses ArticleDownloader to fetch files. Writes to output_dir in the layout
expected by the paper pickler: output_dir/prefix/rest.html (or .xml for Elsevier).

Run separately from the CatalMOF pipeline, e.g.:
  python -m catalmof.text_mining_tools.downloader --doi-csv dois.csv --output-dir /path/to/html/corpus

DOIs in the CSV must be in a column named 'doi'. Optional column 'mode' overrides
publisher detection from DOI prefix. ACS is skipped (violation of ACS policy).
Wiley is supported. For Elsevier DOIs the user must obtain an API key and pass it
via --elsevier-key or CATALMOF_ELSEVIER_API_KEY.
"""

import argparse
import os
import sys

import pandas as pd

from .adjusted_article_downloader.articledownloader import ArticleDownloader

# DOI prefix -> publisher mode (matches Article.split_doi getter_dict).
# Wiley is supported (unlike some legacy downloader scripts).
DOI_PREFIX_TO_MODE = {
    "10.1039": "rsc",
    "10.1002": "wiley",
    "10.1111": "wiley",
    "10.1560": "wiley",
    "10.1562": "wiley",
    "10.1038": "nature",
    "10.1295": "nature",
    "10.1013": "nature",
    "10.1057": "nature",
    "10.1126": "aaas",
    "10.1021": "acs",
    "10.1006": "elsevier",
    "10.1016": "elsevier",
    "10.1529": "elsevier",
    "10.1007": "springer",
    "10.1023": "springer",
    "10.1134": "springer",
    "10.1163": "springer",
}


def _mode_for_doi(doi, mode_from_csv=None):
    if mode_from_csv and str(mode_from_csv).strip().lower() not in ("", "nan"):
        return str(mode_from_csv).strip().lower()
    prefix = doi.split("/", 1)[0].strip()
    return DOI_PREFIX_TO_MODE.get(prefix)


def main():
    parser = argparse.ArgumentParser(
        description="Download article HTML/XML by DOI into the layout expected by the paper pickler."
    )
    parser.add_argument(
        "--doi-csv",
        required=True,
        help="CSV with a 'doi' column (and optional 'mode' column).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory: files written as output_dir/prefix/rest.html or .xml",
    )
    parser.add_argument(
        "--elsevier-key",
        default=os.environ.get("CATALMOF_ELSEVIER_API_KEY"),
        help="API key for Elsevier DOIs (user must obtain separately). Optional env: CATALMOF_ELSEVIER_API_KEY.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=150,
        help="Request timeout in seconds (default 150).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.doi_csv):
        print(f"DOI CSV not found: {args.doi_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.doi_csv)
    if "doi" not in df.columns:
        print("CSV must contain a 'doi' column.", file=sys.stderr)
        sys.exit(1)

    dois = df["doi"].astype(str).str.strip().tolist()
    modes = df["mode"].tolist() if "mode" in df.columns else [None] * len(dois)
    output_dir = os.path.normpath(args.output_dir).rstrip(os.sep)
    os.makedirs(output_dir, exist_ok=True)

    downloader = ArticleDownloader(els_api_key=args.elsevier_key, timeout_sec=args.timeout)

    skipped_acs = 0
    failed = []

    for doi, mode_override in zip(dois, modes):
        if "/" not in doi:
            failed.append((doi, "invalid doi"))
            continue

        mode = _mode_for_doi(doi, mode_override)
        if mode is None:
            failed.append((doi, "unknown publisher prefix"))
            continue
        # ACS skipped: violation of ACS policy.
        if mode == "acs":
            skipped_acs += 1
            continue

        prefix, rest = doi.split("/", 1)
        prefix = prefix.strip()
        rest = rest.strip()
        dir_path = os.path.join(output_dir, prefix)
        os.makedirs(dir_path, exist_ok=True)
        ext = ".xml" if mode == "elsevier" else ".html"
        file_path = os.path.join(output_dir, prefix, rest + ext)

        try:
            with open(file_path, "wb") as f:
                if mode == "elsevier":
                    ok = downloader.get_xml_from_doi(doi, f, mode)
                else:
                    ok = downloader.get_html_from_doi(doi, f, mode)

            if not ok or os.path.getsize(file_path) == 0:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                failed.append((doi, "empty or failed download"))
            else:
                print(doi)
        except Exception as e:
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            failed.append((doi, str(e)))

        try:
            if os.path.isdir(dir_path) and not os.listdir(dir_path):
                os.rmdir(dir_path)
        except OSError:
            pass

    if failed:
        print(f"\nFailed ({len(failed)}):", file=sys.stderr)
        for doi, reason in failed[:20]:
            print(f"  {doi}: {reason}", file=sys.stderr)
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more", file=sys.stderr)
    if skipped_acs:
        print(f"Skipped {skipped_acs} ACS DOI(s) (violation of ACS policy).", file=sys.stderr)
    print(f"Output: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
