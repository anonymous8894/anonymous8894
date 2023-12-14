import collections
import csv
import dataclasses
import json
import multiprocessing
import os
import pickle
import re
import shutil
import subprocess
import traceback
import typing

import numpy as np
import tqdm

import c_embedding
import pycparser
import tracerutils

TMP_FOLDER = "tmp/"
PREPROCESSOR = "gcc"
# PREPROCESSOR = "clang"
FAKE_INCLUDE_PATH = "../../utils/fake_libc_include"
FUNCS_INCLUDE_PATH = "c_include_contain_funcs"
PREPROCESSOR_FLAGS = ["-E", "-I", FUNCS_INCLUDE_PATH, "-I", FAKE_INCLUDE_PATH]
OUTPUT_FOLDER = "checkout/processed/"
PRE_PROCESS_INPUT_FILE_NAME = "__cgrammar_preprocess_input.c"

COMPILER = "gcc"
COMPILER_FLAGS = ["-std=c99", "-c"]
COMPILE_INPUT_FILE_NAME = "__cgrammar_compile_input.c"
COMPILE_OUTPUT_FILE_NAME = "__cgrammar_compile_input.o"

RE_LINEMARKER = re.compile(r"#\s*(\d+)\s+" + r'"([^"]+)"' + r"\s*(\d+(?:\s+\d+)*)?\s*")

NUM_PROCESSES = 64

ST = {
    "ID": "IDENTIFIER",
    "INT_CONST_DEC": "LITERAL_INT",
    "INT_CONST_OCT": "LITERAL_INT",
    "INT_CONST_HEX": "LITERAL_INT",
    "INT_CONST_BIN": "LITERAL_INT",
    "INT_CONST_CHAR": "LITERAL_INT",
    "FLOAT_CONST": "LITERAL_FLOAT",
    "HEX_FLOAT_CONST": "LITERAL_FLOAT",
    "CHAR_CONST": "LITERAL_INT",
    "WCHAR_CONST": "LITERAL_INT",
    "STRING_LITERAL": "LITERAL_STRING",
    "WSTRING_LITERAL": "LITERAL_STRING",
}


# GLOBAL_STAGE = "find_tokens"
GLOBAL_STAGE = "gen_samples"
# GLOBAL_STAGE = None


class ProcessingError(Exception):
    reason: str

    def __init__(self, reason: str):
        self.reason = reason


class ErrorWriter:
    file: typing.TextIO
    name: str

    def __init__(self):
        self.file = None
        self.name = None

    def write(self, text: str):
        if self.file is None:
            self.file = open(self.name, "w")
        self.file.write(text + "\n")

    def new_file_start(self, name: str):
        if self.file is not None:
            self.file.close()
            self.file = None
            self.name = None
        self.name = name


def preprocess_file(file_name: str, err: ErrorWriter):
    pid = os.getpid()
    working_dir = f"{TMP_FOLDER}{pid}/"
    os.makedirs(working_dir, exist_ok=True)
    sample_file_input = f"{working_dir}{PRE_PROCESS_INPUT_FILE_NAME}"
    sample_file_output = f"{working_dir}preprocess_output.c"
    if os.path.exists(sample_file_input):
        os.remove(sample_file_input)
    if os.path.exists(sample_file_output):
        os.remove(sample_file_output)
    shutil.copy(file_name, sample_file_input)
    result = subprocess.run(
        [
            PREPROCESSOR,
            *PREPROCESSOR_FLAGS,
            sample_file_input,
            "-o",
            sample_file_output,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        if err is not None:
            err.write(
                "FAILED TO PROPRESS:\nSTDOUT: {}\nSTDERR: {}".format(
                    result.stdout.decode("utf-8", errors="ignore"),
                    result.stderr.decode("utf-8", errors="ignore"),
                ),
            )
        raise ProcessingError("failed_preprocess")
    if not os.path.exists(sample_file_output):
        if err is not None:
            err.write(
                f"FAILED TO PROPRESS:\nNO OUTPUT FILE",
            )
        raise ProcessingError("failed_preprocess_no_output")
    with open(sample_file_output) as fin:
        code_text = fin.readlines()
    return code_text


def preprocess(
    sample: tracerutils.TracerSample,
    err: ErrorWriter,
    output_folder: str,
    is_original: bool,
):
    if is_original:
        output_original_file = f"{output_folder}original.c"
        code = sample.code
    else:
        output_original_file = f"{output_folder}original_fixed.c"
        code = sample.target
    with open(output_original_file, "w") as fout:
        fout.write(code)
    if is_original:
        if try_compile(output_original_file):
            raise ProcessingError("original_compiles")
    else:
        if not try_compile(output_original_file):
            raise ProcessingError("original_fixed_not_compile")
    code_text = preprocess_file(output_original_file, err)
    if isinstance(code_text, str):
        return code_text
    if is_original:
        output_original_preprocessed_file = f"{output_folder}preprocessed.c"
    else:
        output_original_preprocessed_file = f"{output_folder}preprocessed_fixed.c"
    with open(output_original_file, "w") as fout:
        fout.writelines(code_text)
    return code_text


Token = typing.NamedTuple("Token", [("type", str), ("name", str), ("value", str)])


@dataclasses.dataclass
class EdgeModification:
    insertions: list[Token]
    modification: Token | None
    deletions: bool


@dataclasses.dataclass
class BraceInProgram:
    content: list[Token]
    content_dst: list[Token]
    inserted_loc: int
    matching: list[(Token, Token, str)]
    edge_modifications: list[EdgeModification]


@dataclasses.dataclass
class CodeSegment:
    file: str
    line: int
    flags: list[int]
    begin_line: str
    is_original: bool
    code: list[str]
    code_fixed: list[str]
    tokens: list[Token]
    tokens_fixed: list[Token]
    truncated_tokens: list[Token]
    braces: list[BraceInProgram]
    include_line: typing.Optional[str]
    matching: list[(Token, Token, str)]


def split_code(code_text: list[str]) -> list[CodeSegment]:
    appending = None
    in_original_file = False
    result = []
    known_include_files = {
        "stdio.h",
        "stdlib.h",
        "string.h",
        "math.h",
        "limits.h",
        "strings.h",
        "ctype.h",
        "float.h",
    }
    for line in code_text:
        if line.startswith("#"):
            if appending is not None and (appending.code or appending.include_line):
                result.append(appending)
            match = RE_LINEMARKER.match(line)
            if match is None:
                raise ValueError(f"Failed to parse line marker: {line}")
            line_number = int(match.group(1))
            file_name = match.group(2)
            flags = [int(x) for x in match.group(3).split()] if match.group(3) else []
            is_original = PRE_PROCESS_INPUT_FILE_NAME in file_name
            include_line = None
            if in_original_file and not is_original and file_name != "<built-in>":
                if not file_name.startswith(
                    FAKE_INCLUDE_PATH
                ) and not file_name.startswith(FUNCS_INCLUDE_PATH):
                    print("NOT FILE NAME IN FAKE INCLUDE.", file_name)
                else:
                    if file_name.startswith(FUNCS_INCLUDE_PATH):
                        rel_file_name = file_name[len(FUNCS_INCLUDE_PATH) + 1 :]
                    elif file_name.startswith(FAKE_INCLUDE_PATH):
                        rel_file_name = file_name[len(FAKE_INCLUDE_PATH) + 1 :]
                    else:
                        raise ValueError(f"Invalid file name: {file_name}")
                    if rel_file_name not in known_include_files:
                        print("NEW INCLUDE FILE", rel_file_name)
                    include_line = f'#include "{rel_file_name}"'
            appending = CodeSegment(
                file=file_name,
                line=line_number,
                flags=flags,
                begin_line=line,
                is_original=is_original,
                code=[],
                code_fixed=None,
                tokens=None,
                truncated_tokens=None,
                braces=None,
                include_line=include_line,
                matching=None,
                tokens_fixed=None,
            )
            in_original_file = is_original
        else:
            appending.code.append(line)
    if appending is not None and appending.code:
        result.append(appending)
    return result


def tokenize_lines(code: list[str]) -> list[Token]:
    lexer = pycparser.c_lexer.CLexer(
        lambda x, y, z: None, lambda: None, lambda: None, lambda x: False
    )
    lexer.build()
    code_text = "".join(code)
    lexer.input(code_text)
    tokens = []
    while True:
        next_token = lexer.token()
        if next_token is None:
            break
        next_token_type = next_token.type
        next_token_value = next_token.value
        if next_token_type in ST:
            tokens.append(Token("ST", ST[next_token_type], next_token_value))
        else:
            tokens.append(Token("LT", next_token_value, next_token_value))
    return tokens


def tokenize_segment(code: CodeSegment):
    code.tokens = tokenize_lines(code.code)
    code.tokens_fixed = tokenize_lines(code.code_fixed)


def tokenize_all(segments: list[CodeSegment]) -> set[str]:
    for segment in segments:
        tokenize_segment(segment)


def stat_tokens(segments: list[CodeSegment]):
    sts = set()
    lts = set()
    ids = collections.defaultdict(int)
    for segment in segments:
        for token_list in [segment.tokens, segment.tokens_fixed]:
            for token in token_list:
                if token.type == "ST":
                    if token.name == "IDENTIFIER":
                        if segment.is_original:
                            ids[token.value] += 1
                    else:
                        sts.add(token.name)
                elif token.type == "LT":
                    lts.add(token.name)
    return sts, lts, ids


def find_braces(code: CodeSegment) -> bool:
    if not code.is_original:
        code.truncated_tokens = code.tokens
        code.braces = []
        return

    braces = []
    truncated_tokens = []
    cur_depth = 0
    brace_start = None
    for i, t in enumerate(code.tokens):
        if t.name == "{":
            if cur_depth == 0:
                brace_start = i
                truncated_tokens.append(t)
            cur_depth += 1
        elif t.name == "}":
            cur_depth -= 1
            if cur_depth == 0:
                braces.append(
                    BraceInProgram(
                        content=code.tokens[brace_start + 1 : i],
                        inserted_loc=len(truncated_tokens),
                        matching=None,
                        content_dst=None,
                        edge_modifications=None,
                    )
                )
                truncated_tokens.append(t)
            elif cur_depth < 0:
                if len(braces) == 0:
                    raise ProcessingError("unmatched_brace")
                cur_depth = 0
                last_brace = braces[-1]
                braces[-1].content.extend(truncated_tokens[last_brace.inserted_loc : i])
                truncated_tokens = truncated_tokens[: last_brace.inserted_loc]
                truncated_tokens.append(t)
        else:
            if cur_depth == 0:
                truncated_tokens.append(t)
    if cur_depth != 0:
        braces.append(
            BraceInProgram(
                content=code.tokens[brace_start + 1 :],
                inserted_loc=len(truncated_tokens),
                matching=None,
                content_dst=None,
                edge_modifications=None,
            )
        )
        truncated_tokens.append(Token("LT", "}", "}"))

    code.truncated_tokens = truncated_tokens
    code.braces = braces


def find_braces_all(segments: list[CodeSegment], err: ErrorWriter) -> bool:
    for segment in segments:
        find_braces(segment)


def check_real_func_body(segments: list[CodeSegment], err: ErrorWriter) -> str:
    begin_idxes = []
    token_strs = []
    for i, code in enumerate(segments):
        begin_idxes.append(len(token_strs))
        token_strs.extend([t.value for t in code.truncated_tokens])
    token_strs = "\n".join(token_strs)
    parser = pycparser.CParser()
    try:
        ast = parser.parse(token_strs)
    except pycparser.plyparser.ParseError as e:
        exc = traceback.format_exc()
        err.write(f"FAILED TO PARSE\n{exc}\n{token_strs}")
        raise ProcessingError("failed_to_parse")
    c_ast = pycparser.c_parser.c_ast
    real_func_loc = set()
    for ext in ast.ext:
        if isinstance(ext, c_ast.Typedef):
            pass
        elif isinstance(ext, c_ast.FuncDef):
            func_body_line = ext.body.coord.line
            seg_idx = None
            for i, seg in enumerate(begin_idxes):
                if seg >= func_body_line:
                    seg_idx = i - 1
                    break
            if seg_idx is None:
                seg_idx = len(segments) - 1
            line_in_seg = func_body_line - begin_idxes[seg_idx]
            real_func_loc.add((seg_idx, line_in_seg))
        elif isinstance(ext, c_ast.Decl):
            pass
        else:
            print(type(ext))

    for i, seg in enumerate(segments):
        if not seg.is_original or len(seg.braces) == 0:
            continue
        real_funcs = []
        insert_back = []
        for brace in seg.braces:
            if (i, brace.inserted_loc) in real_func_loc:
                real_funcs.append(brace)
            else:
                insert_back.append(brace)
        if len(insert_back) == 0:
            continue
        truncated_tokens = []
        next_to_insert = 0
        for i in insert_back:
            truncated_tokens.extend(
                seg.truncated_tokens[next_to_insert : i.inserted_loc]
            )
            truncated_tokens.extend(i.content)
            next_to_insert = i.inserted_loc
            for r in real_funcs:
                if r.inserted_loc > i.inserted_loc:
                    r.inserted_loc += len(i.content)
        truncated_tokens.extend(seg.truncated_tokens[next_to_insert:])
        seg.truncated_tokens = truncated_tokens
        seg.braces = real_funcs


def try_compile(file_name: str) -> bool:
    pid = os.getpid()
    working_dir = f"{TMP_FOLDER}{pid}/"
    os.makedirs(working_dir, exist_ok=True)
    compile_input_file_name = f"{working_dir}{COMPILE_INPUT_FILE_NAME}"
    compile_output_file_name = f"{working_dir}{COMPILE_OUTPUT_FILE_NAME}"
    os.system(f"rm -rf {compile_input_file_name} {compile_output_file_name}")

    shutil.copy(file_name, compile_input_file_name)
    result = subprocess.run(
        [
            COMPILER,
            *COMPILER_FLAGS,
            compile_input_file_name,
            "-o",
            compile_output_file_name,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    succ = False
    with open(f"{file_name}.result", "w") as fout:
        if result.returncode == 0:
            if os.path.exists(compile_output_file_name):
                fout.write("RESULT:OK\n")
                succ = True
            else:
                fout.write("RESULT:OK_NO_FILE\n")
        else:
            fout.write("RESULT:FAIL\n")
        stdout = result.stdout.decode("utf-8", errors="ignore")
        stderr = result.stderr.decode("utf-8", errors="ignore")
        fout.write(f"STDOUT:\n{stdout}\n")
        fout.write(f"STDERR:\n{stderr}\n")
    return succ


def token_match(token1: Token, token2: Token):
    return (
        token1.type == token2.type
        and token1.name == token2.name
        and token1.value == token2.value
    )


def match_target(seg: CodeSegment):
    if not seg.is_original:
        return
    original_tokens = []
    is_soft_tokens = []
    next_original_token_id = 0
    brace_locs = []
    for brace in seg.braces:
        is_soft_tokens.extend([False] * (brace.inserted_loc - next_original_token_id))
        original_tokens.extend(
            seg.truncated_tokens[next_original_token_id : brace.inserted_loc]
        )
        brace_begin = len(original_tokens)
        is_soft_tokens.extend([True] * len(brace.content))
        original_tokens.extend(brace.content)
        brace_end = len(original_tokens)
        next_original_token_id = brace.inserted_loc
        brace_locs.append((brace_begin, brace_end))
    is_soft_tokens.extend(
        [False] * (len(seg.truncated_tokens) - next_original_token_id)
    )
    original_tokens.extend(seg.truncated_tokens[next_original_token_id:])

    target_tokens = seg.tokens_fixed

    inf = float("inf")

    scores = [
        [inf for _ in range(len(original_tokens) + 1)]
        for _ in range(len(target_tokens) + 1)
    ]
    record_reasons = [
        ["no" for _ in range(len(original_tokens) + 1)]
        for _ in range(len(target_tokens) + 1)
    ]
    scores[0][0] = 0
    for i in range(1, len(target_tokens) + 1):
        for j in range(1, len(original_tokens) + 1):
            ins_score = (
                scores[i - 1][j] + 1
                if is_soft_tokens[j - 1]
                or (j < len(original_tokens) and is_soft_tokens[j])
                else inf
            )
            replace_score = scores[i - 1][j - 1] + 1 if is_soft_tokens[j - 1] else inf
            del_score = scores[i][j - 1] + 1 if is_soft_tokens[j - 1] else inf
            match_score = (
                scores[i - 1][j - 1]
                if token_match(target_tokens[i - 1], original_tokens[j - 1])
                else inf
            )
            next_val, next_selection = min(
                (ins_score, "ins"),
                (replace_score, "replace"),
                (del_score, "del"),
                (match_score, "match"),
            )
            scores[i][j] = next_val
            record_reasons[i][j] = next_selection

    final_score = scores[-1][-1]
    if final_score >= inf:
        raise ProcessingError("no_target_match")

    aligned_array = []
    i = len(target_tokens)
    j = len(original_tokens)
    while i > 0 or j > 0:
        match record_reasons[i][j]:
            case "ins":
                aligned_array.append(("ins", None, target_tokens[i - 1], None, i - 1))
                i -= 1
            case "replace":
                aligned_array.append(
                    (
                        "replace",
                        original_tokens[j - 1],
                        target_tokens[i - 1],
                        j - 1,
                        i - 1,
                    )
                )
                i -= 1
                j -= 1
            case "del":
                aligned_array.append(("del", original_tokens[j - 1], None, j - 1, None))
                j -= 1
            case "match":
                aligned_array.append(
                    (
                        "match",
                        original_tokens[j - 1],
                        target_tokens[i - 1],
                        j - 1,
                        i - 1,
                    )
                )
                i -= 1
                j -= 1
            case "no":
                assert False
    aligned_array.reverse()
    seg.matching = aligned_array

    last_end = 0
    for i, ((brace_begin, brace_end), brace) in enumerate(zip(brace_locs, seg.braces)):
        aligned_begin, aligned_end = None, None
        for j, (_, _, _, original_idx, _) in enumerate(aligned_array[last_end:]):
            if original_idx == brace_begin - 1:
                aligned_begin = j + last_end + 1
            if original_idx == brace_end:
                aligned_end = j + last_end
                break
        last_end = aligned_end
        brace.matching = aligned_array[aligned_begin:aligned_end]
        tokens_origin = [t for _, t, _, _, _ in brace.matching if t]
        tokens_dst = [t for _, _, t, _, _ in brace.matching if t]
        brace.content_dst = tokens_dst
        assert tuple(tokens_origin) == tuple(brace.content)

        last_insertions = set()
        edge_modifications = [EdgeModification([], None, False)]
        for op, _, target, _, _ in brace.matching:
            match op:
                case "ins":
                    last_insertions.add(target)
                case "del":
                    assert len(last_insertions) == 0
                    edge_modifications.append(
                        EdgeModification(list(last_insertions), None, True)
                    )
                    last_insertions = set()
                case "replace":
                    edge_modifications.append(
                        EdgeModification(list(last_insertions), target, False)
                    )
                    last_insertions = set()
                case "match":
                    edge_modifications.append(
                        EdgeModification(list(last_insertions), None, False)
                    )
                    last_insertions = set()
        edge_modifications.append(EdgeModification(list(last_insertions), None, False))
        brace.edge_modifications = edge_modifications


def match_target_all(segments: list[CodeSegment]):
    for seg in segments:
        match_target(seg)


def gen_samples(
    segments: list[CodeSegment],
) -> "list[c_embedding.TrainingSample]":
    samples = []
    for seg in segments:
        if not seg.is_original:
            continue
        for brace in seg.braces:
            embedding = c_embedding.EmbeddingToken()
            brace.edge_modifications
            token_embeddings = (
                [embedding.embed_begin()]
                + [embedding.embed(token) for token in brace.content]
                + [embedding.embed_end()]
            )
            token_types, token_values = zip(*token_embeddings)
            token_types = np.array(token_types, dtype=np.int16)
            token_values = np.array(token_values, dtype=np.int16)
            mods = [
                (
                    mod.deletions,
                    embedding.embed(mod.modification)[1]
                    if mod.modification is not None
                    else -1,
                    np.array(
                        [embedding.embed(ins)[1] for ins in mod.insertions],
                        dtype=np.int16,
                    ),
                )
                for mod in brace.edge_modifications
            ]
            deletions, modifications, insertions = zip(*mods)
            deletions = np.array(deletions, dtype=np.bool_)
            modifications = np.array(modifications, dtype=np.int16)

            assert len(token_types) == len(token_values)
            assert len(token_types) == len(deletions)
            assert len(token_types) == len(modifications)
            assert len(token_types) == len(insertions)

            uids = embedding.get_uids()

            sample = c_embedding.TrainingSample(
                token_types, token_values, deletions, modifications, insertions, uids
            )
            samples.append(sample)

    return samples


def regenerate_code(
    segments: list[CodeSegment], output_folder: str
) -> list[(str, str)]:
    program_id = 0
    gen_tokens = []
    all_tokens = []
    valid_samples = []
    for i, seg in enumerate(segments):
        if seg.is_original:
            next_to_insert = 0
            for brace in seg.braces:
                new_tokens = list(
                    t.value
                    for t in seg.truncated_tokens[next_to_insert : brace.inserted_loc]
                )
                gen_tokens.extend(new_tokens)
                all_tokens.extend(new_tokens)
                next_to_insert = brace.inserted_loc

                output_env_name = f"{output_folder}env_{program_id}.c"
                with open(output_env_name, "w") as f:
                    f.write("\n".join(gen_tokens))
                    f.write("\n}\n")

                output_tofix_name = f"{output_folder}tofix_{program_id}.c"
                with open(output_tofix_name, "w") as f:
                    for t in brace.content:
                        assert "\n" not in t.value
                        assert "\n" not in t.name
                        assert "\t" not in t.value
                        assert "\t" not in t.name
                    f.write(
                        "\n".join(
                            f"{t.type}\t{t.name}\t{t.value}" for t in brace.content
                        )
                    )

                output_tofix_raw_name = f"{output_folder}tofix_raw_{program_id}.c"
                with open(output_tofix_raw_name, "w") as f:
                    f.write("\n".join(t.value for t in brace.content))

                output_withfunc_name = f"{output_folder}withfunc_{program_id}.c"
                with open(output_withfunc_name, "w") as f:
                    f.write("\n".join(gen_tokens))
                    f.write("\n")
                    f.write("\n".join(t.value for t in brace.content))
                    f.write("\n}\n")
                if not try_compile(output_withfunc_name):
                    valid_samples.append((output_env_name, output_tofix_name))

                all_tokens.extend(t.value for t in brace.content)
                program_id += 1

            new_tokens = list(t.value for t in seg.truncated_tokens[next_to_insert:])
            gen_tokens.extend(new_tokens)
            all_tokens.extend(new_tokens)
        else:
            if seg.include_line is not None:
                gen_tokens.append(seg.include_line)
                all_tokens.append(seg.include_line)
    all_env_name = f"{output_folder}all_env.c"
    with open(all_env_name, "w") as f:
        f.write("\n".join(gen_tokens))
    if not try_compile(all_env_name):
        raise ProcessingError("env_not_compile")
    all_tokens_name = f"{output_folder}all_tokens.c"
    with open(all_tokens_name, "w") as f:
        f.write("\n".join(all_tokens))
    if try_compile(all_tokens_name):
        raise ProcessingError("all_tokens_compile")

    return valid_samples


@dataclasses.dataclass
class ProcessResult:
    succ: bool
    code_id: str
    err_reason: str
    valid_samples: list[(str, str)]


def align_fixed_text(
    segments: list[CodeSegment], segments_fixed: list[CodeSegment]
) -> None:
    if len(segments) != len(segments_fixed):
        raise ProcessingError("num_segs_mismatch")
    for s, sf in zip(segments, segments_fixed):
        if s.file != sf.file:
            raise ProcessingError("seg_file_name_mismatch")
        if s.is_original != sf.is_original:
            raise ProcessingError("seg_is_original_mismatch")
        s.code_fixed = sf.code


def process(sample: tracerutils.TracerSample) -> ProcessResult:
    try:
        err = ErrorWriter()
        output_folder = f"{OUTPUT_FOLDER}{sample.code_id}/"
        os.makedirs(output_folder, exist_ok=True)
        output_error_file = f"{output_folder}err.log"
        err.new_file_start(output_error_file)
        preprocessed = preprocess(sample, err, output_folder, True)
        preprocessed_fixed = preprocess(sample, err, output_folder, False)
        splitted = split_code(preprocessed)
        splitted_fixed = split_code(preprocessed_fixed)
        align_fixed_text(splitted, splitted_fixed)
        tokenize_all(splitted)
        if GLOBAL_STAGE == "find_tokens":
            return stat_tokens(splitted)
        find_braces_all(splitted, err)
        check_real_func_body(splitted, err)
        match_target_all(splitted)
        if GLOBAL_STAGE == "gen_samples":
            samples = gen_samples(splitted)
            sample_file = f"{output_folder}samples.pkl"
            with open(sample_file, "wb") as f:
                pickle.dump(samples, f)
            return ProcessResult(
                succ=True,
                code_id=sample.code_id,
                err_reason="",
                valid_samples=[[sample_file]],
            )
        regen = regenerate_code(splitted, output_folder)
        if len(regen) == 0:
            raise ProcessingError("no_valid_samples")
        return ProcessResult(
            succ=True,
            code_id=sample.code_id,
            err_reason="",
            valid_samples=regen,
        )
    except ProcessingError as e:
        return ProcessResult(
            succ=False,
            code_id=sample.code_id,
            err_reason=e.reason,
            valid_samples=[],
        )


def main():
    os.system(f"rm -rf {OUTPUT_FOLDER}")

    pool = multiprocessing.Pool(processes=NUM_PROCESSES)

    dataset = tracerutils.load_all()
    if GLOBAL_STAGE == "find_tokens":
        lts = set()
        sts = set()
        ids = collections.Counter()
    os.makedirs("checkout", exist_ok=True)
    with open("checkout/result.csv", "w", newline="") as fout:
        writer = csv.writer(fout)
        t = tqdm.tqdm(pool.imap_unordered(process, dataset), total=len(dataset))
        for r in t:
            if GLOBAL_STAGE == "find_tokens":
                if isinstance(r, tuple):
                    sts_, lts_, ids_ = r
                    lts.update(lts_)
                    sts.update(sts_)
                    ids.update(ids_)
                    continue
            writer.writerow(
                (
                    r.code_id,
                    r.succ,
                    r.err_reason,
                    json.dumps([list(x) for x in r.valid_samples]),
                )
            )
            fout.flush()

    if GLOBAL_STAGE == "find_tokens":
        with open("checkout/lts.txt", "w") as f:
            f.write(repr(list(lts)))
        with open("checkout/sts.txt", "w") as f:
            f.write(repr(list(sts)))
        with open("checkout/ids.txt", "w") as f:
            f.write(repr([v for v, _ in ids.most_common(512)]))


if __name__ == "__main__":
    main()
