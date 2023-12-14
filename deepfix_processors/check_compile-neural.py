import multiprocessing
import os
import tqdm
import json

import process_all
import processed_loader


def load_tokens(file_name):
    tokens = []
    with open(file_name) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            tokens.append(line)
    return tokens


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


def save_tokens(file_name, tokens):
    with open(file_name, "w") as fout:
        for line in tokens:
            fout.write(line)
            fout.write("\n")


def process(sample: processed_loader.RunningSample):
    token_file = sample.token_file
    env_file = sample.env_file
    output_file = token_file + "_output_test_neural2"
    ass_file = token_file + ".fix-neural.ass.c"
    run_info_file = token_file + "_runinfo_test_neural2"
    is_success = False
    if os.path.exists(run_info_file) and os.path.exists(output_file):
        with open(run_info_file) as fin:
            for line in fin:
                line = line.strip()
                if line.startswith("---RESULT---"):
                    line = line[13:]
                    line = line.split(",")
                    line = [x.split(":") for x in line]
                    line = [(x[0], x[1]) for x in line]
                    line = dict(line)
                    if "length" in line and line["length"] != "-1":
                        is_success = True
                        break
    if not is_success:
        return None
    env_tokens = load_tokens(env_file)
    output_tokens = process_file_json(output_file)
    assert env_tokens[-1] == "}"
    all_tokens = env_tokens[:-1] + output_tokens + ["}"]
    save_tokens(ass_file, all_tokens)
    if not process_all.try_compile(ass_file):
        return ass_file
    return None


def main():
    samples = list(processed_loader.get_running_samples())
    len_samples = len(samples)
    pool = multiprocessing.Pool(process_all.NUM_PROCESSES)
    with open("checkout/try_compile_neural.txt", "w") as fout:
        for r in tqdm.tqdm(pool.imap_unordered(process, samples), total=len_samples):
            if r is not None:
                print(r)
                fout.write(r)
                fout.write("\n")


if __name__ == "__main__":
    main()
