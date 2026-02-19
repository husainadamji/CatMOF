"""
Build a pickle corpus from a directory of manuscript HTML/XML files.

Expected HTML directory layout: path/prefix/rest.html (or .xml)
  e.g. path/10.1021/acscatal.2c02096.html for DOI 10.1021/acscatal.2c02096

Pickle output: CatalMOF writes to text_mining_dir/pickles (paths.text_mining_pickle_dir).
  e.g. .../pickles/10.1021/acscatal.2c02096.pkl

Reads from config (CATALMOF_CONFIG): paths.text_mining_html_dir. Pickle dir is set by CatalMOF.
"""

import os
import pickle
import sys

from catalmof.paths import get_paths, get_config
from .article import Article


def _discover_dois(html_dir):
    """Traverse html_dir (path/prefix/rest.html|.xml) and yield (doi, prefix, rest) for each file."""
    html_dir = os.path.normpath(html_dir).rstrip(os.sep)
    if not os.path.isdir(html_dir):
        return
    for prefix in os.listdir(html_dir):
        prefix_path = os.path.join(html_dir, prefix)
        if not os.path.isdir(prefix_path):
            continue
        for name in os.listdir(prefix_path):
            base, ext = os.path.splitext(name)
            if ext.lower() not in (".html", ".xml"):
                continue
            yield f"{prefix}/{base}", prefix, base


def pickle_article(doi, html_dir, pickle_dir, skip_existing=True):
    """
    Load HTML/XML for doi from html_dir, run full_analysis, save Article to pickle_dir/prefix/rest.pkl.
    Returns one of: 'pickled', 'skipped', 'error'.
    """
    prefix = doi.split("/", 1)[0]
    rest = doi.split("/", 1)[1]
    pkl_path = os.path.join(pickle_dir, prefix, rest + ".pkl")

    if skip_existing and os.path.isfile(pkl_path):
        return "skipped"

    html_dir_norm = html_dir.rstrip("/") + "/"
    try:
        article = Article(doi, html_dir_norm, download=False)
        article.full_analysis()
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(article, f)
        return "pickled"
    except Exception as e:
        print(f"  Error {doi}: {e}", file=sys.stderr)
        return "error"


def main():
    sys.setrecursionlimit(50000)
    config = get_config()
    p = get_paths()

    html_dir = p.text_mining_html_dir
    if not html_dir or not os.path.isdir(html_dir):
        print("Set paths.text_mining_html_dir in config to your HTML/XML corpus directory.", file=sys.stderr)
        sys.exit(1)

    pickle_dir = os.path.normpath(p.text_mining_pickle_dir).rstrip(os.sep)
    os.makedirs(pickle_dir, exist_ok=True)

    skip_existing = config.get("text_mining_pickler_skip_existing", True)

    count_pickled = count_skipped = count_error = 0
    for doi, prefix, rest in _discover_dois(html_dir):
        result = pickle_article(doi, html_dir, pickle_dir, skip_existing=skip_existing)
        if result == "pickled":
            count_pickled += 1
            print(doi)
        elif result == "skipped":
            count_skipped += 1
        else:
            count_error += 1

    print(f"Done: {count_pickled} pickled, {count_skipped} skipped (existing), {count_error} errors.", file=sys.stderr)
    print(f"Pickles written to: {pickle_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
