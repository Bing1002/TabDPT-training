from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from tqdm import tqdm

from tabdpt_datasets.openml import OpenMLDataset
from model import TabDPTModel, Task
from tabdpt import TabDPTClassifier, TabDPTRegressor
from utils import FAISS, DataPreprocessor


class FullEval:
    def __init__(
        self,
        device,
        max_feat,
        impute_method="mean",
        use_retrieval=False,
    ):
        """Initialize the FullEval class for evaluating tabular data.

        Args:
            device (str): The device to use for evaluation (e.g., "cuda:0" or "cpu").
            max_feat (int): The maximum number of features to use for evaluation.
            impute_method (str, optional): The imputation method to use for missing values. Defaults to "mean".
            use_retrieval (bool, optional): Whether to use retrieval-augmented generation. Defaults to False.
        """
        self.device = device
        self.max_feat = max_feat
        self.use_retrieval = use_retrieval

        # loading classification dataset
        df_eval_cls = pd.read_csv("data_splits/cls_datasets.csv")
        self.cc18_dids = df_eval_cls[df_eval_cls["test_all"] == True][
            "did"
        ].values.tolist()  # 72 datasets

        # loading regression dataset
        reg_df = pd.read_csv("data_splits/reg_datasets.csv")
        ctr_df = reg_df[reg_df["test"] == True]
        self.ctr_dids = ctr_df["did"].values.tolist()

        # get did (dataset ID) to tid (task ID) mapping
        did_tid_mapping = dict(
            zip(
                df_eval_cls[df_eval_cls["test_all"] == True]["did"],
                df_eval_cls[df_eval_cls["test_all"] == True]["tid"],
            )
        )
        did_tid_mapping.update(dict(zip(ctr_df["did"], ctr_df["tid"])))

        self.datasets = {}
        for did in [*self.cc18_dids, *self.ctr_dids]:
            dataset = OpenMLDataset("openml_dataset", openml_task_id=int(did_tid_mapping[did]))
            dataset.prepare_data("data")
            dataset_name = dataset.openml_dataset.name

            X_train, y_train = dataset.train_instances()
            X_val, y_val = dataset.val_instances()

            X_train = np.concatenate([X_train, X_val], axis=0)
            y_train = np.concatenate([y_train, y_val], axis=0)

            X_test, y_test = dataset.test_instances()

            # TODO missing convert_cat2num step in full_dataset.py
            # Preprocess data
            preprocessor = DataPreprocessor(impute_method=impute_method)
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            # Create faiss index
            faiss_knn = FAISS(X_train, use_hnsw=False, metric="L2")

            # back to tensor
            X_train = torch.tensor(X_train).to(device)
            X_test = torch.tensor(X_test).to(device)
            y_train = torch.tensor(y_train).to(device)
            y_test = torch.tensor(y_test)

            self.datasets[did] = (dataset_name, faiss_knn, X_train, X_test, y_train, y_test)

    @torch.no_grad()
    def eval(
        self,
        model: TabDPTModel,
        context_length: int = 1024,
        inf_batch_size=512,
        temperature=0.8,
        return_individual_perfs=False,
    ) -> tuple[dict, Optional[dict]]:
        """Evaluate tdicthe model on classification and regression datasets.

        Args:
            model (TabDPTModel): The TabDPT model to evaluate.
            context_length (int, optional): . Defaults to 1024.
            inf_batch_size (int, optional): Inference batch size. Defaults to 512.
            temperature (float, optional): _description_. Defaults to 0.8.
            return_individual_perfs (bool, optional): _description_. Defaults to False.

        Returns:
            dict: performance metrics for classification and regression tasks.
            Optional[dict]: individual performance metrics for each dataset if return_individual_perfs is True.
        """
        classifier = TabDPTClassifier(
            model=model,
            mode=Task.CLS,
            device=self.device,
            inf_batch_size=inf_batch_size,
            tensor_eval=True,
        )
        regressor = TabDPTRegressor(
            model=model,
            mode=Task.REG,
            device=self.device,
            inf_batch_size=inf_batch_size,
            tensor_eval=True,
        )

        cls_performance = defaultdict(lambda: defaultdict(list))
        reg_performance = defaultdict(lambda: defaultdict(list))

        final_perfs = {}
        individual_perfs = {}

        # evaluation for classification datasets
        for did in tqdm(self.cc18_dids):
            dataset_name, faiss_index, X_train, X_test, y_train, y_test = self.datasets[did]

            classifier.fit(X_train, y_train, faiss_index)
            pred_val = classifier.predict_proba(
                X_test,
                temperature=temperature,
                context_size=context_length,
                use_retrieval=self.use_retrieval,
            )

            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, pred_val[:, 1])
            else:
                auc = roc_auc_score(y_test, pred_val, multi_class="ovo")

            f1 = f1_score(y_test, np.argmax(pred_val, axis=1), average="weighted")
            acc = accuracy_score(y_test, np.argmax(pred_val, axis=1))
            ce = torch.nn.functional.cross_entropy(
                torch.Tensor(pred_val).float(), torch.Tensor(y_test).long()
            )
            cls_performance["cc18"][did] = [acc, f1, auc, ce]
            individual_perfs[dataset_name] = [acc, f1, auc, ce]

        cls_perfs = np.array(list(cls_performance["cc18"].values()))
        cls_perfs_mean = cls_perfs.mean(0)

        final_perfs["cls-cc18-acc"] = cls_perfs_mean[0]
        final_perfs["cls-cc18-f1"] = cls_perfs_mean[1]
        final_perfs["cls-cc18-auc"] = cls_perfs_mean[2]
        final_perfs["cls-cc18-ce"] = cls_perfs_mean[3]

        # evaluation for regression datasets
        for did in self.ctr_dids:
            dataset_name, faiss_index, X_train, X_test, y_train, y_test = self.datasets[did]

            regressor.fit(X_train, y_train, faiss_index)
            pred_val = regressor.predict(
                X_test, context_size=context_length, use_retrieval=self.use_retrieval
            ).flatten()
            # scaler = StandardScaler()
            # y_test = scaler.fit_transform(y_test.reshape(-1, 1)).flatten()

            mse = np.mean((y_test.cpu().numpy() - pred_val) ** 2)
            correlation = scipy.stats.pearsonr(y_test.cpu().numpy(), pred_val.flatten())
            r2 = r2_score(y_test.cpu().numpy(), pred_val)

            reg_performance["ctr"][did] = [mse, correlation[0], r2]
            individual_perfs[dataset_name] = [mse, correlation[0], r2]

        reg_perfs = np.array(list(reg_performance["ctr"].values()))
        reg_perfs_mean = reg_perfs.mean(0)
        final_perfs["reg-ctr-mse"] = reg_perfs_mean[0]
        final_perfs["reg-ctr-cor"] = reg_perfs_mean[1]
        final_perfs["reg-ctr-r2"] = reg_perfs_mean[2]

        if return_individual_perfs:
            return final_perfs, individual_perfs
        else:
            return final_perfs
