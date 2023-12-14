import csv
import json
import os
import subprocess

import nltk
import tqdm

import processed_loader
import pycparser


def tokenize_code(code):
    lexer = pycparser.c_lexer.CLexer(
        lambda x, y, z: None, lambda: None, lambda: None, lambda x: False
    )
    lexer.build()
    lexer.input(code)
    tokens = []
    while True:
        next_token = lexer.token()
        if next_token is None:
            break
        tokens.append(next_token.value)
    return tokens


def format_code(code):
    return [token.strip() for token in code]


def process_json_token(token):
    if token["ty"] == "SymbolicTerminal":
        assert token["value"] != ""
        return token["value"]
    elif token["ty"] == "LiteralTerminal":
        return token["name"]
    else:
        raise Exception(f"Unknown token type {token['ty']}")


def process_file_json(f):
    with open(f) as fin:
        data = json.load(fin)[0]
    result = [process_json_token(token) for token in data]
    return result


def process_file_raw(f):
    with open(f) as fin:
        data = fin.read()
    return tokenize_code(data)


def process_file_strucure(f):
    result = []
    with open(f) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            line = line.split("\t", 3)[2]
            result.append(line)
    return result


def pool_func(x):
    return x[0]


def cmp_func(x, y):
    return nltk.edit_distance(x, y)


def print_code(code):
    code = "\n".join(code)
    p = subprocess.Popen(["clang-format"], stdin=subprocess.PIPE, text=True)
    p.communicate(code)


def main():
    samples = list(processed_loader.get_running_samples())
    with open("checkout/cmp_edit_distance.csv", "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["name", "s1", "s2"])
        for sample in tqdm.tqdm(samples):
            try:
                file0 = sample.token_file
                file1 = sample.token_file + ".fix-dedup.c"
                file2 = sample.token_file + "_output_test_neural2"
                if not os.path.exists(file1) or not os.path.exists(file2):
                    continue
                file0 = format_code(process_file_strucure(file0))
                file1 = format_code(process_file_raw(file1))
                file2 = format_code(process_file_json(file2))
            except Exception:
                continue

            # print("------------------\n")
            # print_code(file0)
            # print("\n\n------------------\n")
            # print_code(file1)
            # print("\n\n------------------\n")
            # print_code(file2)
            # print("\n\n------------------\n")
            # return

            s1 = cmp_func(file0, file1)
            s2 = cmp_func(file0, file2)
            writer.writerow([sample.token_file, s1, s2])


if __name__ == "__main__":
    main()
