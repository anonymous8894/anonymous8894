import dataclasses
import typing

import numpy as np

import process_all

TOKEN_TYPES = [
    "[PAD]",
    "[CLS]",
    "[SEP]",
    "[COPY]",
    "[REMOVE]",
    "[LT]",
    "[ST]",
    "[GLOBAL_ID]",
    "[USER_ID]",
]
SPS = [
    "[PAD]",
    "[CLS]",
    "[SEP]",
    "[COPY]",
    "[REMOVE]",
]
STS = ["LITERAL_FLOAT", "LITERAL_INT", "LITERAL_STRING"]
LTS = [
    "!",
    "!=",
    "#",
    "%",
    "%=",
    "&",
    "&&",
    "(",
    ")",
    "*",
    "*=",
    "+",
    "++",
    "+=",
    ",",
    "-",
    "--",
    "-=",
    "->",
    ".",
    "...",
    "/",
    "/=",
    ":",
    ";",
    "<",
    "<<",
    "<=",
    "=",
    "==",
    ">",
    ">=",
    ">>",
    "?",
    "[",
    "]",
    "^",
    "{",
    "|",
    "|=",
    "||",
    "}",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "unsigned",
    "void",
    "while",
]
GLOBAL_IDS = [
    "pow",
    "getchar",
    "sqrt",
    "fmin",
    "fmax",
    "printf",
    "scanf",
    "malloc",
    "calloc",
    "free",
    "abs",
    "exit",
    "strlen",
]

TOKEN_TYPES_EMBEDDING = {t: i for i, t in enumerate(TOKEN_TYPES)}
SPS_EMBEDDING = {t: i for i, t in enumerate(SPS)}
STS_EMBEDDING = {t: i + len(SPS) for i, t in enumerate(STS)}
LTS_EMBEDDING = {t: i + len(SPS) + len(STS) for i, t in enumerate(LTS)}
GLOBAL_IDS_EMBEDDING = {
    t: i + len(SPS) + len(STS) + len(LTS) for i, t in enumerate(GLOBAL_IDS)
}
USER_ID_START = len(SPS) + len(STS) + len(LTS) + len(GLOBAL_IDS)


class EmbeddingName:
    def __init__(self):
        self.index_to_name = list()
        self.name_to_index = dict()

    def embed(self, name: str):
        if not name in self.name_to_index:
            self.name_to_index[name] = len(self.index_to_name)
            self.index_to_name.append(name)
        return self.name_to_index[name]

    def get_name(self, index):
        return self.index_to_name[index]

    def get_names(self):
        return self.index_to_name


class EmbeddingToken:
    def __init__(self):
        self.embedding_name = EmbeddingName()

    def embed(self, token: "process_all.Token"):
        match token.type:
            case "LT":
                token_type = TOKEN_TYPES_EMBEDDING["[LT]"]
                token_value = LTS_EMBEDDING[token.name]
            case "ST":
                if token.name == "IDENTIFIER":
                    if token.value in GLOBAL_IDS:
                        token_type = TOKEN_TYPES_EMBEDDING["[GLOBAL_ID]"]
                        token_value = GLOBAL_IDS_EMBEDDING[token.value]
                    else:
                        token_type = TOKEN_TYPES_EMBEDDING["[USER_ID]"]
                        token_value = (
                            self.embedding_name.embed(token.value) + USER_ID_START
                        )
                else:
                    token_type = TOKEN_TYPES_EMBEDDING["[ST]"]
                    token_value = STS_EMBEDDING[token.name]
        return token_type, token_value

    def embed_begin(self):
        return (
            TOKEN_TYPES_EMBEDDING["[CLS]"],
            SPS_EMBEDDING["[CLS]"],
        )

    def embed_end(self):
        return (
            TOKEN_TYPES_EMBEDDING["[SEP]"],
            SPS_EMBEDDING["[SEP]"],
        )

    def embed_copy(self):
        return (
            TOKEN_TYPES_EMBEDDING["[COPY]"],
            SPS_EMBEDDING["[COPY]"],
        )

    def embed_remove(self):
        return (
            TOKEN_TYPES_EMBEDDING["[REMOVE]"],
            SPS_EMBEDDING["[REMOVE]"],
        )

    def embed_pad(self):
        return (
            TOKEN_TYPES_EMBEDDING["[PAD]"],
            SPS_EMBEDDING["[PAD]"],
        )

    def get_uids(self):
        return self.embedding_name.get_names()


def get_original(v: int | typing.Tuple[int, int], uids: list[str]):
    if isinstance(v, tuple):
        v = v[1]
    if v < len(SPS):
        return SPS[v]
    elif v < len(SPS) + len(STS):
        return STS[v - len(SPS)]
    elif v < len(SPS) + len(STS) + len(LTS):
        return LTS[v - len(SPS) - len(STS)]
    elif v < len(SPS) + len(STS) + len(LTS) + len(GLOBAL_IDS):
        return GLOBAL_IDS[v - len(SPS) - len(STS) - len(LTS)]
    else:
        return uids[v - len(SPS) - len(STS) - len(LTS) - len(GLOBAL_IDS)]


@dataclasses.dataclass
class TrainingSample:
    token_types: np.ndarray
    token_values: np.ndarray
    deletions: np.ndarray
    modifications: np.ndarray
    insertions: list[np.ndarray]
    uids: list[str]


@dataclasses.dataclass
class FinalTrainingSample:
    inputs: np.ndarray
    outputs: np.ndarray
    insertions: np.ndarray


v_pad = EmbeddingToken().embed_pad()[1]
v_copy = EmbeddingToken().embed_copy()[1]
v_remove = EmbeddingToken().embed_remove()[1]

import c_hyper_parameters as hyper_parameters

num_tokens = USER_ID_START + hyper_parameters.max_identifiers

should_remove = {
    "#",
    "->",
    ".",
    "...",
    "enum",
    "goto",
    "inline",
    "static",
    "struct",
    "typedef",
}
