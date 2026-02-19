import os
import sys
import glob
import pandas as pd
import pickle
import numpy as np
from data.atomic_data import catalysis_kws as default_catalysis_kws
from text_mining_tools.article import Article
from catalmof.paths import get_paths, get_config


def scraper(unique_stable_file, all_records_file):
    """Join manuscript records (name, doi, title) with stable MOF list; return rows with valid doi."""
    all_records_df = pd.read_csv(all_records_file)
    uniq_stable_df = pd.read_csv(unique_stable_file)
    uniq_stable_df = uniq_stable_df.sort_values(by='name').reset_index(drop=True)
    uniq_stable_mofs = uniq_stable_df['name'].values

    matching_indices = []
    for index, row in all_records_df.iterrows():
        temp_name = str(row['name'])
        for uniq_stable_mof in uniq_stable_mofs:
            if temp_name == uniq_stable_mof:
                matching_indices.append(index)
                break
    
    matched_records_df = all_records_df.loc[matching_indices]
    matched_records_df = matched_records_df.sort_values(by='name').reset_index(drop=True)
    final_df = pd.concat([matched_records_df, uniq_stable_df.drop(columns=['name'])], axis=1)
    final_df = final_df[final_df['doi'].notna() & (final_df['doi'] != 'unknown')].reset_index(drop=True)

    return final_df

def load_pickle(base_dir, doi):
    """Load Article pickle for doi from base_dir (prefix/rest.pkl). Returns None if not found."""
    try:
        file_path = glob.glob(f"{base_dir}/{doi}*")[0]
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except IndexError:
        print(f"No file found for DOI: {doi}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"File not found: {file_path}", file=sys.stderr)
        return None


def check_catalysis_in_title(df, catalysis_kws):

    catalysis_in_title = []
    for _, row in df.iterrows():
        title = str(row['title'])
        found_in_title = any(keyword in title for keyword in catalysis_kws)
        catalysis_in_title.append(found_in_title)

    return catalysis_in_title


def check_intro_section(base_dir, df):

    has_intro_section = []
    intro_name_list = []
    for _, row in df.iterrows():
        data = load_pickle(base_dir, row['doi'])
        if data is None:
            has_intro_section.append(0)
            intro_name_list.append('')
        else:
            found_intro = False
            for section_name in data.section_name_dict.items():
                if 'introduction' in section_name[1].lower():
                    found_intro = True
                    has_intro_section.append(1)
                    intro_name_list.append(section_name[1])
                    break
            if not found_intro:
                has_intro_section.append(0)
                intro_name_list.append('')

    return has_intro_section, intro_name_list


def get_full_paper_sentences(base_dir, df, has_intro_section, intro_name_list):

    full_paper_sentences_list = []
    for i, row in df.iterrows():
        data = load_pickle(base_dir, row['doi'])
        if data is None:
            full_paper_sentences_list.append([])
            continue
        if has_intro_section[i]:
            this_paper_sentences = []
            for section_tuple in data.section_text_dict_sentences.items():
                if list(section_tuple[1].keys())[0] == intro_name_list[i]:
                    continue
                this_paper_sentences.extend(list(section_tuple[1].items())[0][1])
        else:
            this_paper_sentences = data.full_paper_sentences

        trimmed_sentences = [
            sentence
            for sentence in this_paper_sentences
            if 'this article references' not in sentence.lower() and 'other publications' not in sentence.lower()
        ]
        full_paper_sentences_list.append(trimmed_sentences)

    return full_paper_sentences_list


def calculate_position_in_paper(full_paper_sentences_list, catalysis_in_title, has_intro_section, catalysis_kws):

    pos_in_paper_vec = []
    for i, sentence_set in enumerate(full_paper_sentences_list):
        if not sentence_set:
            continue
        if not catalysis_in_title[i] and not has_intro_section[i]:
            for word in catalysis_kws:
                for j, sentence in enumerate(sentence_set):
                    if word in sentence.lower():
                        pos_in_paper = float(j) / len(sentence_set)
                        pos_in_paper_vec.append(pos_in_paper)

    return pos_in_paper_vec


def analyze_sentences(full_paper_sentences_list, catalysis_in_title, has_intro_section, catalysis_kws, cutoff):

    have_catalysis_words_list = []
    have_catalysis_words_count = []
    have_catalysis_list = []
    have_catalysis_list_sum = []

    for i, sentence_set in enumerate(full_paper_sentences_list):
        if catalysis_in_title[i]:
            have_catalysis_list.append(True)
            have_catalysis_words_list.append('in title')
            have_catalysis_words_count.append('in title')
            have_catalysis_list_sum.append('in title')
        else:
            catalysis_bool_list, counter = analyze_sentence_set(sentence_set, has_intro_section[i], catalysis_kws, cutoff)
            have_catalysis_list.append(any(catalysis_bool_list))
            have_catalysis_words_list.append(catalysis_bool_list)
            have_catalysis_words_count.append(counter)
            have_catalysis_list_sum.append(sum(catalysis_bool_list))

    return have_catalysis_words_list, have_catalysis_words_count, have_catalysis_list, have_catalysis_list_sum


def analyze_sentence_set(sentence_set, has_intro_section, catalysis_kws, cutoff):

    if not sentence_set:
        return [False] * len(catalysis_kws), [0] * len(catalysis_kws)

    catalysis_bool_list = []
    counter = []
    for word in catalysis_kws:
        count = 0
        for j, sentence in enumerate(sentence_set):
            if word in sentence.lower() and (has_intro_section or float(j) / len(sentence_set) > cutoff):
                count += 1
        counter.append(count)
        catalysis_bool_list.append(count > 0)

    return catalysis_bool_list, counter


def _get_keywords():
    """Build text-mining keyword list from config: merge or override with default."""
    config = get_config()
    user_keywords = config.get("text_mining_keywords", [])
    if not isinstance(user_keywords, list):
        user_keywords = []
    mode = config.get("text_mining_keywords_mode", "merge")
    if mode == "override":
        return list(user_keywords) if user_keywords else list(default_catalysis_kws)
    # merge: default first, then user extras, dedupe by order
    seen = set()
    out = []
    for w in list(default_catalysis_kws) + user_keywords:
        w = str(w).strip()
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def main():
    p = get_paths()
    config = get_config()
    keywords = _get_keywords()

    if not os.path.isfile(p.manuscript_data_csv):
        raise FileNotFoundError(
            "Text mining requires a CSV with MOF names, DOIs, and titles. "
            "Set paths.manuscript_data_csv in config to your file (columns: name, doi, title). "
            "You can create it using: python -m catalmof.text_mining_tools.title_fetcher --doi-csv <doi.csv> --output-csv <out.csv>"
        )

    # Default to title-only (True) unless user explicitly sets False AND provides pickle directory
    title_only_config = config.get("text_mining_title_only", True)
    pickle_dir = p.text_mining_pickle_dir
    # If no pickle directory provided, force title-only mode
    if pickle_dir is None:
        title_only = True
    else:
        title_only = title_only_config

    uniq_stable_mof_manuscript_df = scraper(
        p.stable_mofs_unique_mc_csv,
        p.manuscript_data_csv,
    )

    catalysis_in_title = check_catalysis_in_title(uniq_stable_mof_manuscript_df, keywords)
    uniq_stable_mof_manuscript_df["catalysis_in_title"] = catalysis_in_title

    if title_only:
        # Binary classification: any keyword in title or not. No full-paper analysis.
        uniq_stable_mof_manuscript_df.to_csv(p.text_mining_results_csv, index=False)
        print(uniq_stable_mof_manuscript_df["catalysis_in_title"].value_counts())
        uniq_stable_nocat_mof_df = uniq_stable_mof_manuscript_df.loc[
            ~uniq_stable_mof_manuscript_df["catalysis_in_title"]
        ].drop(columns=["catalysis_in_title"])
        uniq_stable_nocat_mof_df.to_csv(p.stable_uniq_no_catalysis_csv, index=False)
        return

    # Full text mining: intro, full paper sentences, position, sentence analysis
    if pickle_dir is None:
        raise ValueError(
            "Full paper text mining requires text_mining_pickle_dir in config paths. "
            "Set paths.text_mining_pickle_dir to your pickle file directory, or use title-only mode."
        )

    cutoff = config.get("text_mining_cutoff", 0.25)
    max_catalysis_hits = config.get("text_mining_max_catalysis_hits", 3)

    has_intro_section, intro_name_list = check_intro_section(pickle_dir, uniq_stable_mof_manuscript_df)
    full_paper_sentences_list = get_full_paper_sentences(
        pickle_dir, uniq_stable_mof_manuscript_df, has_intro_section, intro_name_list
    )

    pos_in_paper_vec = calculate_position_in_paper(
        full_paper_sentences_list, catalysis_in_title, has_intro_section, keywords
    )
    pos_in_paper_df = pd.DataFrame({"position_in_paper": pos_in_paper_vec})
    pos_in_paper_df.to_csv(p.position_in_paper_csv, index=False)

    analysis_results = analyze_sentences(
        full_paper_sentences_list, catalysis_in_title, has_intro_section, keywords, cutoff
    )
    (
        have_catalysis_words_list,
        have_catalysis_words_count,
        have_catalysis_list,
        have_catalysis_list_sum,
    ) = analysis_results

    uniq_stable_mof_manuscript_df["has_intro_section"] = has_intro_section
    uniq_stable_mof_manuscript_df["have_catalysis_words"] = have_catalysis_words_list
    uniq_stable_mof_manuscript_df["have_catalysis_words_count"] = have_catalysis_words_count
    uniq_stable_mof_manuscript_df["have_catalysis"] = have_catalysis_list
    uniq_stable_mof_manuscript_df["have_catalysis_sum"] = have_catalysis_list_sum

    uniq_stable_mof_manuscript_df.to_csv(p.text_mining_results_csv, index=False)
    print(uniq_stable_mof_manuscript_df["have_catalysis"].value_counts())

    # Retain MOFs with at most (max_catalysis_hits - 1) unique catalysis keyword hits.
    # "in title" (catalysis_in_title) is excluded; only numeric have_catalysis_sum are compared.
    ser = pd.to_numeric(uniq_stable_mof_manuscript_df["have_catalysis_sum"], errors="coerce")
    uniq_stable_nocat_mof_df = uniq_stable_mof_manuscript_df.loc[
        (ser.notna()) & (ser < max_catalysis_hits)
    ]
    uniq_stable_nocat_mof_df = uniq_stable_nocat_mof_df.drop(
        columns=[
            "catalysis_in_title",
            "has_intro_section",
            "have_catalysis_words",
            "have_catalysis_words_count",
            "have_catalysis",
            "have_catalysis_sum",
        ]
    )
    uniq_stable_nocat_mof_df.to_csv(p.stable_uniq_no_catalysis_csv, index=False)


if __name__ == "__main__":
    main()

