from __future__ import annotations

import difflib
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy

from tabdpt_datasets.annotated_tables import AnnotatedTablesDataset
from tabdpt_datasets.dataset import Dataset
from tabdpt_datasets.openml import (
    AMLBDataset,
    CC18Dataset,
    CTR23Dataset,
    GrinsztajnDataset,
    ICLRTrainingDataset,
    TabZillaDataset,
)
from tabdpt_datasets.tabred import TabredDataset
from tabdpt_datasets.talent import TalentDataset


class NpGenericEncoder(json.JSONEncoder):
    # Handle types like np.float32
    def default(self, x):
        if isinstance(x, np.generic):
            return x.item()
        return json.JSONEncoder.default(self, x)


# Only include dataset classes that provide a suite of named datasets that can each be loaded by
# name
SUITES = [
    CC18Dataset,
    CTR23Dataset,
    GrinsztajnDataset,
    TabredDataset,
    TalentDataset,
    AMLBDataset,
    TabZillaDataset,
    ICLRTrainingDataset,
    AnnotatedTablesDataset,
]

EVAL_SUITES = [CC18Dataset, CTR23Dataset]
EVAL_SUITE_NAMES = [s.suite_name() for s in EVAL_SUITES]


def longest_overlap(s1: str, s2: str) -> int:
    sm = difflib.SequenceMatcher(a=s1.lower(), b=s2.lower())
    return sm.find_longest_match(0, len(s1), 0, len(s2))[2]


@dataclass
class Duplicate:
    suite_name_1: str
    dataset_name_1: str
    suite_name_2: str
    dataset_name_2: str
    likelihood: Literal["low", "high", "certain"]
    reasons: list[str]

    def likelihood_gte(self, other: str):
        """
        Returns True if this duplicate's likelihood is greater than or equal to the other given
        likelihood.
        """
        assert other in ["low", "high", "certain"]
        return (
            other == "low"
            or other == "high"
            and self.likelihood != "low"
            or other == "certain"
            and self.likelihood == "certain"
        )


class CatalogueView:
    """
    Base class for catalogue functions that don't manage metadata on disk. Allows for filtered views
    of the catalogue.

    Code outside this module generally shouldn't directly instantiate CatalogueViews, the Catalogue
    class should be used instead.
    """

    def __init__(self, metadata):
        self._metadata = metadata
        self.dataset_map = {(c.suite_name(), name): c for c in SUITES for name in c.all_names()}

    def filter(self, key: str, pred) -> CatalogueView:
        """
        Filter catalogue using a metadata key and corresponding required value, or a metadata key
        and a predicate that must return True when passed the value.

        E.g., c.filter('target_type', 'regression') or c.filter('size', lambda s: s > 1000)

        Returns a new CatalogueView.
        """
        if callable(pred):
            return CatalogueView({k: v for k, v in self._metadata.items() if pred(v[key])})
        return CatalogueView({k: v for k, v in self._metadata.items() if v[key] == pred})

    def metadata(self):
        return self._metadata

    def dataset(self, suite_name: str, dataset_name: str) -> Dataset:
        """
        Get a Dataset object for the dataset with the given name.
        """
        return self.dataset_map[(suite_name, dataset_name)](dataset_name)

    def datasets(self) -> list[Dataset]:
        """
        Get dataset objects for all datasets in the current view.
        """
        return [self.dataset(*k) for k in self._metadata.keys()]

    def detect_duplicates(self, verbose=False) -> list[Duplicate]:
        """
        Detects possible duplicate datasets. Sensitivity can be selected for detecting leakage (high
        sensitivity, default) or just removing redundant data (low sensitivity). Returns a list of
        Duplicate objects.

        verbose: Print out duplicates and info on detection as they're detected
        """
        entries = list(self._metadata.items())
        kd_trees = {}
        for k, m in entries:
            col_coeffs = []
            for mean, var, skew, kurt in zip(
                m["column_means"], m["column_vars"], m["column_skews"], m["column_kurtoses"]
            ):
                col_coeffs.append([mean, np.sqrt(var), skew, kurt])
            # Using moderate values for infinities so they don't drive all other values to zero once
            # normalized
            col_coeffs = np.nan_to_num(np.array(col_coeffs), posinf=1.0, neginf=-1.0)
            if len(col_coeffs) > 0:
                # Normalize row-wise so that the distance tolerance works reasonably
                col_coeffs = (col_coeffs - col_coeffs.mean(axis=1)[:, None]) / (
                    col_coeffs.std(axis=1)[:, None] + 1e-8
                )
                kd_trees[k] = scipy.spatial.KDTree(col_coeffs)

        duplicates = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                k1, m1 = entries[i]
                k2, m2 = entries[j]
                # These suites should already be deduped
                if m1["suite_name"] == m2["suite_name"] and m1["suite_name"] in (
                    "cc18",
                    "ctr23",
                    "grinsztajn",
                    "tabred",
                    "amlb",
                ):
                    continue

                dup_reasons = []
                certain_dup = False
                high_feature_similarity = False

                if m1["n_rows"] == m2["n_rows"] and sum(c != "0" for c in str(m1["n_rows"])) > 1:
                    # Ignore n_rows mismatches if they only have one non-zero digit, e.g., 10000 vs
                    # 10000 is not really suspicious
                    dup_reasons.append(f"same n_rows {m1['n_rows']}")

                if m1["n_features"] == m2["n_features"]:
                    dup_reasons.append(f"same number of features {m1['n_features']}")

                if (
                    m1["target_type"] != "none"
                    and m2["target_type"] != "none"
                    and np.isclose(m1["y_mean"], m2["y_mean"])
                    and np.isclose(m1["y_var"], m2["y_var"])
                ):
                    dup_reasons.append(
                        f"similar target statistics: mean {m1['y_mean']} var {m1['y_var']} "
                        f"vs mean {m2['y_mean']} var {m2['y_var']}"
                    )

                if "file_sha1" in m1 and "file_sha1" in m2 and m1["file_sha1"] == m2["file_sha1"]:
                    certain_dup = True
                    dup_reasons.append("same sha1 hash")

                m1_name = m1.get("openml_dataset_name", m1.get("kaggle_dataset_name", None))
                m2_name = m2.get("openml_dataset_name", m2.get("kaggle_dataset_name", None))
                if m1_name and m2_name and longest_overlap(m1_name, m2_name) > 4:
                    dup_reasons.append("overlap in names:" f"{m1_name}, {m2_name}")

                # Compare statistics for each column wrt the target. If enough are similar, then
                # there might be leakage.
                if k1 in kd_trees and k2 in kd_trees:
                    # Check the number of columns that seem to have a pair in the other table
                    # Annoying output format - list of lists of indices in other tree
                    pair_inds = kd_trees[k1].query_ball_tree(kd_trees[k2], 1e-3)
                    n_similar_k1 = sum(len(xs) > 0 for xs in pair_inds)
                    n_similar_k2 = len(set(x for xs in pair_inds for x in xs))
                    n_similar = min(n_similar_k1, n_similar_k2)
                    thresh = min(5, m1["n_features"], m2["n_features"])

                    if n_similar >= thresh:
                        dup_reasons.append(f"{n_similar} columns with similar statistics")
                        min_n_features = min(m1["n_features"], m2["n_features"])
                        max_n_features = max(m1["n_features"], m2["n_features"])
                        n_feature_ratio = max_n_features / min_n_features
                        detection_frac = min(1, 1 / 2 ** (np.log10(min_n_features) - 1))
                        if n_similar > detection_frac * min_n_features and n_feature_ratio < 1.5:
                            high_feature_similarity = True

                likelihood = (
                    "certain"
                    if certain_dup
                    else (
                        "high"
                        if len(dup_reasons) > 2
                        else "low" if len(dup_reasons) > 1 or high_feature_similarity else None
                    )
                )
                if likelihood:
                    if verbose:
                        print(f"Suspected duplicate {k1} {k2}: {', '.join(dup_reasons)}")
                        if m1["suite_name"] == m2["suite_name"]:
                            print(f"Within same suite! {m1['suite_name']}")
                    duplicates.append(
                        Duplicate(k1[0], k1[1], k2[0], k2[1], likelihood, dup_reasons)
                    )

        return duplicates

    def filter_duplicates(
        self, eval_min_likelihood="low", train_min_likelihood="high"
    ) -> CatalogueView:
        """
        Returns a new CatalogueView with duplicate datasets removed. Eval datasets are always kept.

        eval_min_likelihood: Either 'low', 'high', 'certain', or None. Determines minimum duplicate
            likelihood between a train and eval dataset for the train dataset to be removed. Set to None
            to ignore eval duplicates. The default is 'low' to prevent all suspected leakage.
        train_min_likelihood: Either 'low', 'high', 'certain', or None. Determines minimum duplicate
            likelihood between two train datasets for the first train dataset to be removed. Set to None
            to ignore train duplicates. The default is 'high' to avoid removing useful data, since
            leakage isn't a concern in the train set.
        """
        # Remove all duplicates with eval datasets, then go back and remove the first dataset from
        # each remaining pair of duplicates. Since detect_duplicates uses a fixed ordering, this
        # ensures all duplicates are resolved.
        duplicates = self.detect_duplicates(verbose=False)
        to_remove: set[tuple[str, str]] = set()
        dups_no_eval = []
        for d in duplicates:
            if d.suite_name_1 in EVAL_SUITE_NAMES and d.suite_name_2 in EVAL_SUITE_NAMES:
                continue
            elif d.suite_name_1 in EVAL_SUITE_NAMES:
                if eval_min_likelihood and d.likelihood_gte(eval_min_likelihood):
                    to_remove.add((d.suite_name_2, d.dataset_name_2))
            elif d.suite_name_2 in EVAL_SUITE_NAMES:
                if eval_min_likelihood and d.likelihood_gte(eval_min_likelihood):
                    to_remove.add((d.suite_name_1, d.dataset_name_1))
            else:
                dups_no_eval.append(d)
        for d in dups_no_eval:
            if (
                train_min_likelihood
                and d.likelihood_gte(train_min_likelihood)
                and (d.suite_name_2, d.dataset_name_2) not in to_remove
            ):
                to_remove.add((d.suite_name_1, d.dataset_name_1))
        return CatalogueView({k: v for k, v in self._metadata.items() if k not in to_remove})

    def total_size(self) -> (int, int):
        """
        Total instances and cells across all datasets
        """
        return (
            sum(d["n_rows"] for d in self._metadata.values()),
            sum(d["n_cells"] for d in self._metadata.values()),
        )

    def split(self) -> tuple[CatalogueView, CatalogueView]:
        """
        Splits datasets based on name into a training and eval set. Note that to avoid leakage,
        duplicates should already be filtered out. Uses fixed splits as discussed.
        """
        train, eval = {}, {}
        for k, v in self._metadata.items():
            if k[0] in EVAL_SUITE_NAMES:
                eval[k] = v
            else:
                train[k] = v
        return CatalogueView(train), CatalogueView(eval)


class Catalogue(CatalogueView):
    """
    Dataset catalogue, used to access the aggregate set of datasets in our project and filter,
    remove duplicates, split into train and hold-out, etc.

    Unlike CatalogueView, this class includes functionality to generate, load, and store the
    catalogue, so it should be generally used to create and access the catalogue.
    """

    def __init__(self, download_dir, suites=SUITES):
        self.download_dir = download_dir
        self.suites = suites
        self.cache_path = os.path.join(download_dir, "metadata.json")
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                try:
                    metadata = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    print(
                        f"JSON decoding failed - solution is probably to delete {self.cache_path} and rerun full_update"
                    )
                    raise e
                # Stored as a list, convert back into a dict indexed by suite and dataset name
                metadata = {(d["suite_name"], d["dataset_name"]): d for d in metadata}
        else:
            metadata = {}
            warnings.warn(
                "Catalogue cache doesn't exist yet - run catalogue.incremental_update() or catalogue.full_update() before using"
            )
        super().__init__(metadata)

    def update_dataset(self, dataset, update_cache=True):
        """
        Run auto_populate_metadata to update the metadata for an individual dataset, optionally
        writing to the cache.
        """
        dataset.prepare_data(self.download_dir)
        dataset.auto_populate_metadata()
        self._metadata[(dataset.suite_name(), dataset.name)] = dataset.metadata
        if update_cache:
            with open(self.cache_path, "w") as f:
                json.dump(list(self._metadata.values()), f, cls=NpGenericEncoder, indent=4)

    def incremental_update(self):
        """
        Update metadata and cache for datasets that don't already appear in metadata.
        """
        n_updated = 0
        n_total = sum(len(c.all_names()) for c in self.suites)
        for c in self.suites:
            for name in c.all_names():
                n_updated += 1
                sys.stdout.write(f"\rUpdating catalogue... {n_updated}/{n_total}")
                sys.stdout.flush()
                if name not in self._metadata:
                    self.update_dataset(c(name), update_cache=False)
        print("Done")
        with open(self.cache_path, "w") as f:
            json.dump(list(self._metadata.values()), f, cls=NpGenericEncoder, indent=4)

    def full_update(self):
        """
        Re-generate metadata and cache entirely.
        """
        self._metadata = {}
        n_updated = 0
        n_total = sum(len(c.all_names()) for c in self.suites)
        for c in self.suites:
            for name in c.all_names():
                n_updated += 1
                sys.stdout.write(f"\rUpdating catalogue... {n_updated}/{n_total}")
                sys.stdout.flush()
                self.update_dataset(c(name), update_cache=False)
        print("Done")
        with open(self.cache_path, "w") as f:
            json.dump(list(self._metadata.values()), f, cls=NpGenericEncoder, indent=4)
