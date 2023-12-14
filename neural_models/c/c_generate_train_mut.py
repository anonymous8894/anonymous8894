import csv
import json
import pickle

import numpy as np
import tqdm

import c_embedding
import c_hyper_parameters as hyper_parameters

import random


def process_token(token: int) -> int:
    if token < c_embedding.USER_ID_START:
        return token
    cur = token - c_embedding.USER_ID_START
    cur = cur % hyper_parameters.max_identifiers
    return cur + c_embedding.USER_ID_START


def process_token_list(tokens: list) -> list:
    return [process_token(x) for x in tokens]


def empty_modify(sample: "c_embedding.TrainingSample"):
    l = len(sample.token_values)
    sample.deletions = np.zeros([l], dtype=np.bool_)
    sample.modifications = np.zeros([l], dtype=np.int64) - 1
    sample.insertions = [np.zeros([0], dtype=np.int64) for _ in range(l)]


def random_token(r: random.Random):
    return r.randint(len(c_embedding.SPS), c_embedding.num_tokens - 1)


def mut_sample(
    sample: "c_embedding.TrainingSample", r: random.Random
) -> "c_embedding.TrainingSample":
    mut_type = r.randint(0, 2)
    match mut_type:
        case 0:
            # modify
            if len(sample.token_values) <= 2:
                return sample
            loc = r.randint(1, len(sample.token_values) - 2)
            token = random_token(r)
            if sample.deletions[loc] or sample.modifications[loc] >= 0:
                return sample
            result = c_embedding.TrainingSample(
                None,
                sample.token_values.copy(),
                sample.deletions.copy(),
                sample.modifications.copy(),
                [x for x in sample.insertions],
                None,
            )
            result.modifications[loc] = result.token_values[loc]
            result.token_values[loc] = token
            return result
        case 1:
            # insertion
            loc = r.randint(1, len(sample.token_values) - 1)
            token = random_token(r)
            if (
                sample.insertions[loc - 1].shape[0] > 0
                or sample.insertions[loc].shape[0] > 0
            ):
                return sample
            result_token_values = np.concatenate(
                [sample.token_values[:loc], [token], sample.token_values[loc:]]
            )
            result_deletions = np.concatenate(
                [sample.deletions[:loc], [True], sample.deletions[loc:]]
            )
            result_modifications = np.concatenate(
                [sample.modifications[:loc], [-1], sample.modifications[loc:]]
            )
            result_insertions = (
                sample.insertions[:loc]
                + [np.zeros([0], dtype=np.int64)]
                + sample.insertions[loc:]
            )
            result = c_embedding.TrainingSample(
                None,
                result_token_values,
                result_deletions,
                result_modifications,
                result_insertions,
                None,
            )
            return result
        case 2:
            # deletion
            if len(sample.token_values) <= 3:
                return sample
            loc = r.randint(1, len(sample.token_values) - 2)
            if (
                sample.deletions[loc - 1]
                or sample.deletions[loc]
                or sample.deletions[loc + 1]
            ):
                return sample
            result_token_values = np.concatenate(
                [sample.token_values[:loc], sample.token_values[loc + 1 :]]
            )
            result_deletions = np.concatenate(
                [sample.deletions[:loc], sample.deletions[loc + 1 :]]
            )
            result_modifications = np.concatenate(
                [sample.modifications[:loc], sample.modifications[loc + 1 :]]
            )
            result_insertions = sample.insertions[:loc] + sample.insertions[loc + 1 :]
            result_insertions[loc] = np.concatenate(
                [
                    sample.insertions[loc],
                    sample.insertions[loc + 1],
                    [sample.token_values[loc]],
                ]
            )
            result = c_embedding.TrainingSample(
                None,
                result_token_values,
                result_deletions,
                result_modifications,
                result_insertions,
                None,
            )
            return result


def gen_mut_samples(sample: c_embedding.TrainingSample, mut_seed: int):
    r = random.Random(mut_seed)
    num_gen = 50
    empty_modify(sample)
    for _ in range(num_gen):
        mut_num = r.randint(1, 5)
        for _ in range(mut_num):
            sample = mut_sample(sample, r)
        yield sample


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
    with open("checkout/result_pretraining.csv") as fin:
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
    pkl_files.sort()
    samples = []
    idx = 0
    for pkl_file in tqdm.tqdm(pkl_files):
        with open(pkl_file, "rb") as fin:
            data = pickle.load(fin)
            for x in data:
                idx += 1
                for sample in gen_mut_samples(x, idx):
                    samples.append(process_sample(sample))
    len_samples = len(samples)
    len_valid = int(len_samples * 0.1)
    len_test = int(len_samples * 0.1)
    len_train = len_samples - len_valid - len_test
    train_samples = samples[:len_train]
    valid_samples = samples[len_train : len_train + len_valid]
    test_samples = samples[len_train + len_valid :]
    with open("checkout/c-train-pretraining.pkl", "wb") as fout:
        pickle.dump(train_samples, fout)
    with open("checkout/c-valid-pretraining.pkl", "wb") as fout:
        pickle.dump(valid_samples, fout)
    with open("checkout/c-test-pretraining.pkl", "wb") as fout:
        pickle.dump(test_samples, fout)


if __name__ == "__main__":
    main()
