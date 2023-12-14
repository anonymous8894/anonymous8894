import subprocess
import json
import os
import torch
import csv
import tqdm

import transformers

import processed_loader


def format_code(code):
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
    result = "\n".join(result)
    return result


def process_file_raw(f):
    with open(f) as fin:
        data = fin.read()
    return data


def process_file_strucure(f):
    result = []
    with open(f) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            line = line.split("\t", 3)[2]
            result.append(line)
    result = "\n".join(result)
    return result


def pool_func(x):
    return x[0]


def cmp_func(x, y):
    return torch.cosine_similarity(x, y, dim=0)


def run_model(tokenizer, model, code):
    inputs = tokenizer.tokenize(code)
    tokens = [tokenizer.cls_token] + inputs + [tokenizer.sep_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    inputs = torch.tensor(
        tokens_ids,
        device=model.device,
    )[None, :]
    outputs = model(inputs)[0][0]
    outputs = pool_func(outputs)
    return outputs


def main():
    with torch.no_grad():
        samples = list(processed_loader.get_running_samples())
        device = "cuda:0"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "microsoft/codebert-base"
        )
        model = transformers.AutoModel.from_pretrained("microsoft/codebert-base")
        model.to(device)
        with open("checkout/cmp.csv", "w", newline="") as fout:
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

                v1 = run_model(tokenizer, model, file0)
                v2 = run_model(tokenizer, model, file1)
                v3 = run_model(tokenizer, model, file2)

                s1 = cmp_func(v1, v2)
                s2 = cmp_func(v1, v3)

                s1 = s1.cpu().item()
                s2 = s2.cpu().item()

                writer.writerow([sample.token_file, s1, s2])


if __name__ == "__main__":
    main()
