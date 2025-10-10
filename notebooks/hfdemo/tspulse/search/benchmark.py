import argparse

import faiss
import numpy as np
import pandas as pd
import torch
from datautils import (
    RetrievedData,
    SyntheticCombDataset,
    TransformedSyntheticTestDataset,
    UCRDataset,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

from tsfm_public.models.tspulse import TSPulseForReconstruction
from tsfm_public.models.tspulse.utils.helpers import get_embeddings


@torch.no_grad()
def retrieve(device, batch_size, k, train_dataset, test_dataset, model):
    # compute embeddings of training data
    dataloader = DataLoader(train_dataset, batch_size=batch_size)
    train_embeddings, metadata = [], []
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        embs = get_embeddings(model, batch["past_values"].to(device))
        B, C, D = embs.shape
        metadata.extend([(batch_idx * batch_size + b, c) for b in range(B) for c in range(C)])
        train_embeddings.append(embs.cpu().numpy())
    train_embeddings = np.concatenate(train_embeddings).squeeze(axis=1)  # due to univariate

    # create index set of embeddings from training data
    d = train_embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, 20)
    index.train(train_embeddings)
    index.add(train_embeddings)
    index.nprobe = 5

    # find top-k similar embeedings for query test data
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    D_all, I_all = [], []
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        test_embedding = get_embeddings(model, batch["past_values"].to(device))
        query_vector = test_embedding.squeeze(dim=1).cpu()  # due to univariate, [B, 1, D] -> [B, D]
        D, I = index.search(query_vector, k=k)
        D_all.extend(D)
        I_all.extend(I)

    return D_all, I_all, metadata


def compute_ranking_score(device, batch_size, k, I_all, train_dataset, test_dataset, level):
    def _precision_k(cmp, k):
        cmp_k = np.sum(cmp[:, :k], axis=1) / k
        mean_cmp_k = np.mean(cmp_k)
        return mean_cmp_k

    def _avg_prec_k(cmp, n_rel, k):
        rank = np.arange(k) + 1
        cmp_k = np.cumsum(cmp[:, :k], axis=1)
        cmp_k = cmp_k / rank
        cmp_k = np.sum(cmp_k * cmp[:, :k], axis=1) / np.minimum(n_rel, k)
        mean_cmp_k = np.mean(cmp_k)
        return mean_cmp_k

    def _dcg_k(cmp, k):
        cmp_k = cmp[:, :k]
        n_data = cmp_k.shape[0]
        dcgs = np.zeros(n_data)
        for i in range(n_data):
            idx = np.argwhere(cmp_k[i, :] > 0).flatten()
            dcg = np.reciprocal(np.log2(idx + 2))
            dcgs[i] = np.sum(dcg) / k
        return dcgs

    def _idcg_k(cmp, n_rel, k):
        n_data = cmp.shape[0]
        ideal_cmp = np.zeros((n_data, k))
        n_rel_each = np.minimum(k, n_rel)
        mask = np.repeat((np.arange(k) + 1)[:, None].T, n_data, axis=0)
        mask = mask <= n_rel_each[:, None]
        ideal_cmp[mask] = 1
        return _dcg_k(ideal_cmp, k)

    def _mrr(cmp, k):
        cmp_k = cmp[:, :k]
        first_index = np.argmax(cmp_k, axis=1, keepdims=True)
        first_value = np.take_along_axis(cmp_k, first_index, axis=1)  # false if there is no correct item
        rr = 1 / (first_index + 1)
        return np.mean(rr * first_value)

    retrieveddata = RetrievedData(train_dataset, test_dataset, I_all, level)
    dataloader = DataLoader(retrieveddata, batch_size=batch_size)
    cmp, n_rel = [], []
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        label_test = batch["label_test"]
        labels_train = batch["labels_train"]
        cmp.append(label_test[:, None] == labels_train)
        n_rel.append(batch["n_rel"])
    cmp = np.concatenate(cmp, axis=0)
    n_rel = np.concatenate(n_rel, axis=0)

    # compute metrics
    prec_k = _precision_k(cmp, k)
    avg_prec_k = _avg_prec_k(cmp, n_rel, k)
    dcg_k = _dcg_k(cmp, k)
    idcg_k = _idcg_k(cmp, n_rel, k)
    ndcg_k = np.mean(dcg_k / idcg_k)
    mrr_k = _mrr(cmp, k)

    return prec_k, avg_prec_k, ndcg_k, mrr_k


def evaluate(device, seed, batch_size, k, model, train_dataset, test_dataset):
    D_all, I_all, metadata = retrieve(device, batch_size, k, train_dataset, test_dataset, model)
    df = []
    for level in ["family_match", "finegrained_match"]:
        prec_k, avg_prec_k, ndcg_k, mrr_k = compute_ranking_score(
            device, batch_size, k, I_all, train_dataset, test_dataset, level
        )
        df.append(
            pd.DataFrame(
                {
                    "PREC@k": prec_k,
                    "AP@k": avg_prec_k,
                    "NDCG@k": ndcg_k,
                    "MRR@k": mrr_k,
                    "k": k,
                    "seed": seed,
                    "level": level,
                },
                index=[0],
            )
        )

    df = pd.concat(df, axis=0).reset_index(drop=True)

    return df


def main(seed, batch_size, k, dataset):
    set_seed(seed)
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    if dataset == "synth":
        train_dataset = SyntheticCombDataset()
    elif dataset == "real":
        train_dataset = UCRDataset()
    else:
        raise RuntimeError(f"{dataset} is not found")

    batch_size = min([batch_size, len(train_dataset)])
    test_dataset = TransformedSyntheticTestDataset(train_dataset, max_shift=0.2, max_scale=0.2, noise_ratio=0.1)

    model = TSPulseForReconstruction.from_pretrained(
        "ibm-granite/granite-timeseries-tspulse-r1",
        revision="tspulse-hybrid-dualhead-512-p8-r1",
        num_input_channels=1,
        mask_type="user",
    )
    model.eval().to(device)

    df_result = evaluate(device, seed, batch_size, k, model, train_dataset, test_dataset)
    return df_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["synth", "real"])
    args = parser.parse_args()

    dataset = args.dataset
    batch_size = 1024
    k = 3
    seed = 42
    df_result = main(seed, batch_size, k, dataset)
    df_result.to_csv(f"results_{dataset}.csv")
    print(df_result[["level", "PREC@k", "MRR@k", "AP@k", "NDCG@k"]])
