import argparse
import base64
import itertools
import json
import os

import javalang
import numpy as np

import mjtokens
import model
import splitutil

PREFIXES = "CLASS_", "VAR_", "FIELD_", "METHOD_", "NEW_IDENTIFIER_"


def load_terminals():
    with open(os.path.join(os.path.dirname(__file__), "mj-symbols.json")) as fin:
        j = json.load(fin)
        return j["symbolic_terminals"], j["literal_terminals"]


def map_token(token):
    if token.startswith("__new_id_"):
        token = token.replace("__new_id_", "NEW_IDENTIFIER_")
    return token


def read_tokens(file_name):
    return [
        map_token(x.value) for x in javalang.tokenizer.tokenize(open(file_name).read())
    ]


def to_token_ids(tokens):
    tokens = [mjtokens.tokens_map[x] for x in tokens]
    # tokens = np.array(tokens, dtype=np.int64)
    return tokens


def pad_to_len(vals, length, pad):
    if len(vals) < length:
        vals = vals + [pad] * (length - len(vals))
    else:
        vals = vals[:length]
    vals = np.array(vals, dtype=np.int64)
    return vals


def replace_tokens(tokens, other_tokens):
    maps = []
    for _ in PREFIXES:
        maps.append((set(), list()))
    for token in itertools.chain(tokens, other_tokens):
        for i, prefix in enumerate(PREFIXES):
            if token.startswith(prefix):
                if token not in maps[i][0]:
                    maps[i][0].add(token)
                    maps[i][1].append(token)
                break
    replaces = {}
    for i, prefix in enumerate(PREFIXES):
        for j, token in enumerate(maps[i][1]):
            replaces[token] = f"{prefix}{j}"
    tokens = to_token_ids(replaces.get(x, x) for x in tokens)

    inputs = pad_to_len(
        [mjtokens.v_start] + tokens + [mjtokens.v_end],
        len(tokens) + 2,
        mjtokens.v_pad,
    )

    mappings = dict(zip(replaces.values(), replaces.keys()))

    return inputs, mappings


def all_tokens(tokens, max_new_id, mappings):
    _, literal_terminals = load_terminals()
    literal_terminals = set(literal_terminals)
    all_ids = []
    remove_tokens = {
        "super",
        "class",
        "METHODFIX",
        "extends",
        "Object",
        "CLSFIX",
        "void",
    }
    for token in set(tokens):
        if token in literal_terminals:
            continue
        if token in remove_tokens:
            continue
        if token in mappings:
            tokenid = [mjtokens.tokens_map[mappings[token]]]
        else:
            tokenid = mjtokens.get_range(token)
            tokenid = [mjtokens.tokens_map[x] for x in tokenid if x not in mappings]
        all_ids.append((tokenid, token))
    new_ids = [
        ([mjtokens.tokens_map[f"NEW_IDENTIFIER_{i}"]], f"__new_id_{i}")
        for i in range(max_new_id)
    ]
    all_ids = all_ids + new_ids

    tokens = []
    ids = []
    for lt in literal_terminals:
        tokens.append({"is_symbolic": False, "name": lt, "value": ""})
        ids.append([mjtokens.tokens_map[lt]])
    for id_idx, id_name in all_ids:
        tokens.append({"is_symbolic": True, "name": "IDENTIFIER", "value": id_name})
        ids.append(id_idx)

    return tokens, ids


def process_mj_file(input_file, env_file, output_file, max_len, trainer, max_new_id, weight_splits=10):
    input_tokens = read_tokens(input_file)
    length = len(input_tokens)
    env_tokens = read_tokens(env_file)
    input_tokens_processed, mappings = replace_tokens(input_tokens, env_tokens)
    tokens, ids = all_tokens(
        itertools.chain(input_tokens, env_tokens), max_new_id, mappings
    )
    number_tokens = len(tokens)
    outputs, insertions = trainer.run_instance(input_tokens_processed)
    outputs = -outputs
    insertions = np.max(insertions, 1)
    insertions = -insertions
    outputs_split = splitutil.SplitUtil(outputs.min(), outputs.max(), weight_splits)
    insertions_split = splitutil.SplitUtil(insertions.min(), insertions.max(), weight_splits)
    outputs = outputs_split.get_split(outputs) + 1
    insertions = insertions_split.get_split(insertions) + 1

    insertions[insertions > max_len] = max_len + 1
    outputs[outputs > max_len] = max_len + 1

    origin = np.zeros([length], dtype=np.int16)
    insert = np.ones([length + 1, number_tokens], dtype=np.int16)
    for loc in range(length + 1):
        for i, idx in enumerate(ids):
            minval = max_len + 1
            for j in idx:
                minval = min(minval, insertions[loc, j])
            insert[loc, i] = minval
    update = np.ones([length, number_tokens], dtype=np.int16)
    for loc in range(length):
        for i, idx in enumerate(ids):
            minval = max_len + 1
            for j in idx:
                minval = min(minval, outputs[loc + 1, j])
            update[loc, i] = minval
    remove = np.ones([length], dtype=np.int16)
    for loc in range(length):
        remove[loc] = outputs[loc + 1, mjtokens.v_remove]
    origin, insert, update, remove = map(
        lambda x: base64.b64encode(x.tobytes()).decode("ascii"),
        (origin, insert, update, remove),
    )

    j = {
        "length": length,
        "tokens": tokens,
        "origin": origin,
        "insert": insert,
        "remove": remove,
        "update": update,
    }

    with open(output_file, "w") as fout:
        json.dump(j, fout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("env_file")
    parser.add_argument("output_file")
    parser.add_argument("max_len", type=int)
    arg = parser.parse_args()
    if arg.max_len <= 0:
        print("max_len must be greater than or equal to 0")
    if arg.max_len > 10:
        print("max_len must be less than or equal to 10")
    trainer = model.ModifyingModelTrainer(load_dataset=False, load_writer=False)
    trainer.load_model()
    process_mj_file(arg.input_file, arg.env_file, arg.output_file, arg.max_len, trainer)


if __name__ == "__main__":
    main()
