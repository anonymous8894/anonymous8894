import csv
import json
import multiprocessing
import pickle
import random
import subprocess

import numpy as np
import tqdm

import c_embedding


def random_token(r: random.Random, sample: "c_embedding.TrainingSample"):
    return r.randint(
        len(c_embedding.SPS), c_embedding.USER_ID_START + len(sample.uids) - 1
    )


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
            token = random_token(r, sample)
            result = c_embedding.TrainingSample(
                None,
                sample.token_values.copy(),
                None,
                None,
                None,
                sample.uids,
            )
            result.token_values[loc] = token
            return result
        case 1:
            # insertion
            loc = r.randint(1, len(sample.token_values) - 1)
            token = random_token(r, sample)
            result_token_values = np.concatenate(
                [sample.token_values[:loc], [token], sample.token_values[loc:]]
            )
            result = c_embedding.TrainingSample(
                None,
                result_token_values,
                None,
                None,
                None,
                sample.uids,
            )
            return result
        case 2:
            # deletion
            if len(sample.token_values) <= 3:
                return sample
            loc = r.randint(1, len(sample.token_values) - 2)
            result_token_values = np.concatenate(
                [sample.token_values[:loc], sample.token_values[loc + 1 :]]
            )
            result = c_embedding.TrainingSample(
                None,
                result_token_values,
                None,
                None,
                None,
                sample.uids,
            )
            return result


def gen_mut_samples(sample: c_embedding.TrainingSample, r: random.Random):
    mut_num = r.randint(1, 5)
    for _ in range(mut_num):
        sample = mut_sample(sample, r)
    return sample


def format_code(code):
    code = [c_embedding.get_original(t, code.uids) for t in code.token_values[1:-1]]
    code = "\n".join(code)
    p = subprocess.Popen(
        ["clang-format"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    code = p.communicate(input=code)[0]
    if p.returncode != 0:
        raise Exception("clang-format failed")
    return code


def process_sample(
    sample: c_embedding.TrainingSample,
    r: random.Random,
) -> list[tuple[str, int]]:
    origin = sample
    muted = gen_mut_samples(sample, r)
    origin = (format_code(origin), 1)
    muted = (format_code(muted), 0)
    return [origin, muted]


def process_wrapper(x):
    pkl_id, pkl_file = x
    result = []
    r = random.Random(pkl_id)
    with open(pkl_file, "rb") as fin:
        data = pickle.load(fin)
        for x in data:
            result.extend(process_sample(x, r))
    return result


def main():
    pkl_files = []
    pool = multiprocessing.Pool(16)
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
    for r in tqdm.tqdm(
        pool.imap_unordered(process_wrapper, enumerate(pkl_files)), total=len(pkl_files)
    ):
        samples.extend(r)

    len_samples = len(samples)
    len_valid = int(len_samples * 0.1)
    len_test = int(len_samples * 0.1)
    len_train = len_samples - len_valid - len_test
    train_samples = samples[:len_train]
    valid_samples = samples[len_train : len_train + len_valid]
    test_samples = samples[len_train + len_valid :]
    with open("checkout/c-train-compare.pkl", "wb") as fout:
        pickle.dump(train_samples, fout)
    with open("checkout/c-valid-compare.pkl", "wb") as fout:
        pickle.dump(valid_samples, fout)
    with open("checkout/c-test-compare.pkl", "wb") as fout:
        pickle.dump(test_samples, fout)


if __name__ == "__main__":
    main()
