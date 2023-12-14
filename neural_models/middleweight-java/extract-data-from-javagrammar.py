import itertools
import os
import pickle
import random
import re
import traceback
import json

import javalang
import numpy as np
import tqdm
from nltk.metrics.distance import edit_distance_align

import hyper_parameters
import mjtokens

RE_FIXINFO = re.compile(r"---RESULT---,.*,length:(\d+),.*")

PREFIXES = "CLASS_", "VAR_", "FIELD_", "METHOD_", "NEW_IDENTIFIER_"


def pad_to_len(vals, length, pad):
    if len(vals) < length:
        vals = vals + [pad] * (length - len(vals))
    else:
        vals = vals[:length]
    vals = np.array(vals, dtype=np.int64)
    return vals


def replace_tokens(tokens1, tokens2):
    maps = []
    for _ in PREFIXES:
        maps.append(set())
    for token in itertools.chain(tokens1, tokens2):
        for i, prefix in enumerate(PREFIXES):
            if token.startswith(prefix):
                maps[i].add(token)
                break
    replaces = {}
    for i, prefix in enumerate(PREFIXES):
        for j, token in enumerate(maps[i]):
            replaces[token] = f"{prefix}{j}"
    tokens1, tokens2 = (
        to_token_ids(replaces.get(x, x) for x in tokens1),
        to_token_ids(replaces.get(x, x) for x in tokens2),
    )

    insertions = [set() for _ in range(len(tokens1) + 1)]
    outputs = [None for _ in tokens1]

    for (i1, j1), (i2, j2) in itertools.pairwise(edit_distance_align(tokens1, tokens2)):
        if i1 == i2:
            # insert
            assert j2 == j1 + 1
            insertions[i1].add(tokens2[j1])
        elif j1 == j2:
            # delete
            assert i2 == i1 + 1
            outputs[i1] = mjtokens.v_remove
        else:
            # replace or equals
            assert i2 == i1 + 1
            assert j2 == j1 + 1
            if tokens1[i1] == tokens2[j1]:
                outputs[i1] = mjtokens.v_copy
            else:
                outputs[i1] = tokens2[j1]

    for o in outputs:
        if o is None:
            raise Exception("Invalid output")
    insertions = [
        pad_to_len(sorted(x), hyper_parameters.num_insertions, mjtokens.v_remove)
        for x in insertions
    ]
    insertions = pad_to_len(
        insertions,
        hyper_parameters.sequence_length,
        np.zeros(hyper_parameters.num_insertions, dtype=np.int64) + mjtokens.v_pad,
    )
    outputs = pad_to_len(
        [mjtokens.v_start] + outputs + [mjtokens.v_end],
        hyper_parameters.sequence_length,
        mjtokens.v_pad,
    )
    inputs = pad_to_len(
        [mjtokens.v_start] + tokens1 + [mjtokens.v_end],
        hyper_parameters.sequence_length,
        mjtokens.v_pad,
    )

    return inputs, outputs, insertions


def to_token_ids(tokens):
    tokens = [mjtokens.tokens_map[x] for x in tokens]
    # tokens = np.array(tokens, dtype=np.int64)
    return tokens


def map_token(token):
    if token.startswith("__new_id_"):
        token = token.replace("__new_id_", "NEW_IDENTIFIER_")
    return token


def read_tokens(file_name):
    return [
        map_token(x.value) for x in javalang.tokenizer.tokenize(open(file_name).read())
    ]

def read_tokens_json(file_name):
    with open(file_name) as fin:
        tokens = json.load(fin)[0]
    tokens = [x["value"] if x["ty"] == "SymbolicTerminal" else x["name"] for x in tokens]
    tokens = [map_token(x) for x in tokens]
    return tokens

total_found = 0
total_number = 0


def update_desc(t):
    global total_found, total_number
    t.set_description(
        f"{total_found}/{total_number}, {total_found/(total_number + (0 if total_number else 1))*100:05.02f}%"
    )


def stat(base):
    global total_found, total_number
    for mutant_type in ["p", "pi", "i", "a"]:
        for mutant_count in range(1, 11):
            subfolder = os.path.join(base, f"m_{mutant_type}_{mutant_count}")
            if not os.path.exists(subfolder):
                continue

            samples = os.listdir(subfolder)
            samples = tqdm.tqdm(samples)
            update_desc(samples)
            for sample in samples:
                update_desc(samples)
                total_number += 1
                sample_folder = os.path.join(subfolder, sample)

                block_file = os.path.join(sample_folder, "block")
                env_file = os.path.join(sample_folder, "env")
                output_file = os.path.join(sample_folder, "output_test")
                runinfo_file = os.path.join(sample_folder, "runinfo_test")

                if not (
                    os.path.exists(block_file)
                    and os.path.exists(env_file)
                    and os.path.exists(runinfo_file)
                    and os.path.exists(output_file)
                ):
                    continue

                try:
                    input_tokens = read_tokens(block_file)
                    output_tokens = read_tokens_json(output_file)
                    num_modifications = -1
                    for line in open(runinfo_file):
                        match = RE_FIXINFO.match(line)
                        if match:
                            num_modifications = int(match.group(1))

                    input_tokens, output_tokens, insertions = replace_tokens(
                        input_tokens, output_tokens
                    )
                    if num_modifications > 0:
                        total_found += 1
                        yield sample_folder, input_tokens, output_tokens, insertions
                except Exception:
                    traceback.print_exc()


def main():
    result = list(stat("data"))
    random.shuffle(result)
    num_all = len(result)
    num_test = int(num_all * 0.2)
    num_train = num_all - num_test - num_test
    train = result[:num_train]
    valid = result[num_train + num_test :]
    test = result[num_train : num_train + num_test]

    with open("train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open("valid.pkl", "wb") as f:
        pickle.dump(valid, f)
    with open("test.pkl", "wb") as f:
        pickle.dump(test, f)


if __name__ == "__main__":
    main()
