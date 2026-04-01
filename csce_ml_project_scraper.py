# Common
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# additional for Neural Network
# include MLP or CNN
import tensorflow as tf
from pandas.errors import ParserError
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
# requires pip install ucimlrepo
from ucimlrepo import fetch_ucirepo

# dataset = fetch_ucirepo(id=1)
# print(dataset.metadata)

import argparse
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup



# use_dataset_loader = True



BASE_URL = (
    "https://archive.ics.uci.edu/datasets"
    "?Task=Classification"
    "&Types=Multivariate"
    "&Types=Tabular"
    "&Python=true"
    "&skip=0"
    "&take=10"
    "&sort=desc"
    "&orderBy=NumHits"
    "&search="
)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)


def build_page_url(base_url: str, *, skip: int, take: int) -> str:
    parsed = urlparse(base_url)

    # Keep repeated query params like Types=...
    params = parse_qsl(parsed.query, keep_blank_values=True)

    new_params = []
    skip_set = False
    take_set = False

    for key, value in params:
        if key == "skip":
            new_params.append(("skip", str(skip)))
            skip_set = True
        elif key == "take":
            new_params.append(("take", str(take)))
            take_set = True
        else:
            new_params.append((key, value))

    if not skip_set:
        new_params.append(("skip", str(skip)))
    if not take_set:
        new_params.append(("take", str(take)))

    new_query = urlencode(new_params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def extract_total_count(page_text: str) -> int | None:
    match = re.search(r"\b\d+\s+to\s+\d+\s+of\s+(\d+)\b", page_text)
    return int(match.group(1)) if match else None


def parse_dataset_links(html: str, page_url: str) -> list[int]:
    soup = BeautifulSoup(html, "html.parser")

    anchors = soup.select("a[href*='/dataset/']")
    dataset_ids = []
    seen = set()

    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href:
            continue

        full_url = urljoin(page_url, href)

        match = re.search(r"/dataset/(\d+)", full_url)
        if not match:
            continue

        dataset_id = int(match.group(1))

        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        dataset_ids.append(dataset_id)

    return dataset_ids


def scrape_all_datasets(base_url: str, *, take: int = 25, timeout: int = 30) -> list[int]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_ids = []
    seen_ids = set()
    skip = 0
    total_count = None

    while True:
        page_url = build_page_url(base_url, skip=skip, take=take)

        response = session.get(page_url, timeout=timeout)
        response.raise_for_status()

        html = response.text
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)

        if total_count is None:
            total_count = extract_total_count(text)
            print(f"Total count reported: {total_count}")

        page_ids = parse_dataset_links(html, page_url)
        new_ids = [dataset_id for dataset_id in page_ids if dataset_id not in seen_ids]

        if not new_ids:
            break

        for dataset_id in new_ids:
            seen_ids.add(dataset_id)
            all_ids.append(dataset_id)

        if total_count is not None and len(all_ids) >= total_count:
            break

        if len(page_ids) < take:
            break

        skip += take

    return all_ids


def get_data() -> list[int]:
    parser = argparse.ArgumentParser(
        description="Scrape UCI dataset IDs matching the filtered URL."
    )
    parser.add_argument("--url", default=BASE_URL, help="Filtered UCI listing URL to scrape.")
    parser.add_argument(
        "--take",
        type=int,
        default=25,
        help="Rows requested per page while paginating."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds."
    )
    args = parser.parse_args([])

    ids = scrape_all_datasets(args.url, take=args.take, timeout=args.timeout)

    if not ids:
        raise SystemExit("No dataset IDs were found. The page structure may have changed.")

    return ids

def run():
    # extract ids to exclude based on characertistics
    exclude_types: list[str] = ["Time-Series", "Image", "Sequential", "Spatiotemporal", "Text", "Other"]

    ids = get_data()
    print(ids)

    excluded_ids = []
    final_ids = []

    for dataset_id in ids:
        try:
            dataset = fetch_ucirepo(id=dataset_id)
            y = dataset.data.targets

            if y is None or y.empty:
                print(f"Skipping dataset {dataset_id} ({dataset.metadata['name']}) due to no target variables.")
                excluded_ids.append(dataset_id)
                continue

            # get min class size
            min_class_size = y.value_counts().min()

            if min_class_size < 2:
                print(f"Class size < 2 for {dataset_id}")
                excluded_ids.append(dataset_id)
                continue

            final_ids.append(dataset_id)
        except ParserError as e:
            print(f"Skipping {dataset_id} due to parse error: {e}")
            excluded_ids.append(dataset_id)
            continue
        except Exception as e:
            print(f"Skipping {dataset_id} due to fetch error: {e}")
            excluded_ids.append(dataset_id)
            continue

        characteristics = dataset.metadata.get("characteristics") or []

        if any(exclude in characteristics for exclude in exclude_types):
            print(f"Skipping {dataset_id} due to characteristics: {characteristics}")
            excluded_ids.append(dataset_id)
            continue

    print("Failed dataset count:", len(excluded_ids))

    return final_ids

# if __name__=="__main__":
    # ids = run()