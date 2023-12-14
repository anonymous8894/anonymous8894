import csv
import json
import pickle

import numpy as np
import tqdm

import c_embedding
import c_hyper_parameters as hyper_parameters


def process_token(token: int) -> int:
    if token < c_embedding.USER_ID_START:
        return token
    cur = token - c_embedding.USER_ID_START
    cur = cur % hyper_parameters.max_identifiers
    return cur + c_embedding.USER_ID_START


def process_token_list(tokens: list) -> list:
    return [process_token(x) for x in tokens]


def process_sample(
    sample: c_embedding.TrainingSample,
) -> c_embedding.FinalTrainingSample:
    embed = c_embedding.EmbeddingToken()
    v_remove = embed.embed_remove()[1]
    v_pad = embed.embed_pad()[1]
    v_copy = embed.embed_copy()[1]

    insertions = sample.insertions
    inputs = sample.token_values

    outputs = [
        v_remove if removal else (process_token(mod) if mod >= 0 else v_copy)
        for (mod, removal) in zip(sample.modifications, sample.deletions)
    ]

    insertions = [
        pad_to_len(
            sorted(process_token_list(x)), hyper_parameters.num_insertions, v_remove
        )
        for x in insertions
    ]
    insertions = pad_to_len(
        insertions,
        hyper_parameters.sequence_length,
        np.zeros(hyper_parameters.num_insertions, dtype=np.int64) + v_pad,
    )
    outputs = pad_to_len(
        outputs,
        hyper_parameters.sequence_length,
        v_pad,
    )
    inputs = pad_to_len(
        process_token_list(inputs),
        hyper_parameters.sequence_length,
        v_pad,
    )
    return c_embedding.FinalTrainingSample(inputs, outputs, insertions)


def pad_to_len(vals, length, pad):
    if len(vals) < length:
        vals = list(vals) + [pad] * (length - len(vals))
    else:
        vals = vals[:length]
    vals = np.array(vals, dtype=np.int64)
    return vals


def main():
    pkl_files = []
    with open("checkout/result.csv") as fin:
        reader = csv.reader(fin)
        for _, _, _, f in tqdm.tqdm(reader):
            f = f.strip()
            if len(f) == 0:
                continue
            f = json.loads(f)
            if len(f) == 0:
                continue
            f = f[0][0]
            pkl_files.append(f)
    samples = []
    for pkl_file in tqdm.tqdm(pkl_files):
        with open(pkl_file, "rb") as fin:
            data = pickle.load(fin)
            data = [process_sample(x) for x in data]
            samples.extend(data)
    len_samples = len(samples)
    len_valid = int(len_samples * 0.1)
    len_test = int(len_samples * 0.1)
    len_train = len_samples - len_valid - len_test
    train_samples = samples[:len_train]
    valid_samples = samples[len_train : len_train + len_valid]
    test_samples = samples[len_train + len_valid :]
    with open("checkout/c-train.pkl", "wb") as fout:
        pickle.dump(train_samples, fout)
    with open("checkout/c-valid.pkl", "wb") as fout:
        pickle.dump(valid_samples, fout)
    with open("checkout/c-test.pkl", "wb") as fout:
        pickle.dump(test_samples, fout)


if __name__ == "__main__":
    main()
