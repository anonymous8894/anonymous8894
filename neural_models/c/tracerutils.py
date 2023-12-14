import typing

import pandas as pd

TracerSample = typing.NamedTuple(
    "TracerSample",
    [
        ("code_id", str),
        ("code", str),
        ("target", str),
    ],
)

TRACER_FILE_NAME = "../../utils/tracer/data/dataset/singleL/singleL_Train+Valid.csv"


def load_iter() -> typing.Iterable[TracerSample]:
    df = pd.read_csv(TRACER_FILE_NAME)
    for i in range(len(df)):
        code_id = "train-" + str(df["Unnamed: 0"][i])
        code = df["sourceText"][i]
        target = df["targetText"][i]
        yield TracerSample(
            code_id=code_id,
            code=code,
            target=target,
        )


def load_all() -> list[TracerSample]:
    return list(load_iter())
