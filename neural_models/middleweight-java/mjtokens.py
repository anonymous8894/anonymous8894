t_pad = "#PAD"
t_start = "#START"
t_end = "#END"
t_split = "#SPLIT"
t_unk = "#UNK"
t_cooy = "#COPY"
t_remove = "#REMOVE"

tokens = (
    [
        t_pad,
        t_start,
        t_end,
        t_split,
        t_unk,
        t_cooy,
        t_remove,
        "(",
        ")",
        ",",
        ".",
        ";",
        "=",
        "==",
        "CLSFIX",
        "METHODFIX",
        "Object",
        "else",
        "if",
        "new",
        "null",
        "return",
        "{",
        "}",
    ]
    + [f"CLASS_{x}" for x in range(10)]
    + [f"FIELD_{x}" for x in range(10)]
    + [f"METHOD_{x}" for x in range(10)]
    + [f"NEW_IDENTIFIER_{x}" for x in range(10)]
    + [f"VAR_{x}" for x in range(30)]
)

prefixes = [
    ("CLASS_", 10),
    ("FIELD_", 10),
    ("METHOD_", 10),
    ("NEW_IDENTIFIER_", 10),
    ("VAR_", 10),
]

tokens_map = {x: i for i, x in enumerate(tokens)}
num_tokens = len(tokens)


v_pad = tokens_map[t_pad]
v_start = tokens_map[t_start]
v_end = tokens_map[t_end]
v_split = tokens_map[t_split]
v_unk = tokens_map[t_unk]
v_copy = tokens_map[t_cooy]
v_remove = tokens_map[t_remove]


def get_range(token):
    for p, m in prefixes:
        if token.startswith(p):
            return [f"{p}{x}" for x in range(m)]
    return None


if __name__ == "__main__":
    print(num_tokens)
