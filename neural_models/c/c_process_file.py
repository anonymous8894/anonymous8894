import base64
import json

import numpy as np

import c_embedding
import c_generate_train
import c_hyper_parameters as hyper_parameters
import process_all
import splitutil
import re

RE_ENV = re.compile(r"^=(?:FN|VAR) ([a-zA-Z0-9_]*):.*$")


def parse_cenv(c_env):
    with open(c_env, "r") as fin:
        names = set()
        for line in fin:
            line = line.strip()
            if not line:
                continue
            m = RE_ENV.match(line)
            if m:
                name = m.group(1)
                names.add(name)
    names = sorted(names)
    return names


def process_file(c_file, c_env, trainer, max_len, output_file, weight_splits=10):
    env_names = parse_cenv(c_env)

    input_tokens = []
    with open(c_file, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ty, name, value = line.split("\t", 3)
            token = process_all.Token(ty, name, value)
            input_tokens.append(token)

    embedding = c_embedding.EmbeddingToken()
    token_embeddings = (
        [embedding.embed_begin()[1]]
        + [embedding.embed(token)[1] for token in input_tokens]
        + [embedding.embed_end()[1]]
    )
    token_embeddings = c_generate_train.process_token_list(token_embeddings)
    token_embeddings = np.array(token_embeddings)

    for name in env_names:
        embedding.embed(process_all.Token("ST", "IDENTIFIER", name))
    for i in range(max_len):
        embedding.embed(process_all.Token("ST", "IDENTIFIER", f"__new_id_{i}"))

    tokens = []
    ids = []

    for st in c_embedding.STS:
        match st:
            case "LITERAL_FLOAT":
                value = "0.0"
            case "LITERAL_INT":
                value = "0"
            case "LITERAL_STRING":
                value = '""'
        tokens.append({"is_symbolic": True, "name": st, "value": value})
        ids.append([c_embedding.STS_EMBEDDING[st]])
    for lt in c_embedding.LTS:
        if lt in c_embedding.should_remove:
            continue
        tokens.append({"is_symbolic": False, "name": lt, "value": ""})
        ids.append([c_embedding.LTS_EMBEDDING[lt]])
    for i, name in enumerate(embedding.embedding_name.index_to_name):
        tokens.append({"is_symbolic": True, "name": "IDENTIFIER", "value": name})
        ids.append([i % hyper_parameters.max_identifiers + c_embedding.USER_ID_START])

    number_tokens = len(tokens)

    length = len(token_embeddings) - 2
    outputs, insertions = trainer.run_instance(token_embeddings)

    outputs = -outputs
    insertions = np.max(insertions, 1)
    insertions = -insertions
    outputs_split = splitutil.SplitUtil(outputs.min(), outputs.max(), weight_splits)
    insertions_split = splitutil.SplitUtil(
        insertions.min(), insertions.max(), weight_splits
    )
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
                minval = min(minval, insertions[loc + 1, j])
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
        remove[loc] = outputs[loc + 1, c_embedding.v_remove]
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
