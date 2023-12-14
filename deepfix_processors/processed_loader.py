import csv
import dataclasses
import json
from collections.abc import Iterable


@dataclasses.dataclass
class RunningSample:
    env_file: str
    token_file: str


def get_running_samples() -> Iterable[RunningSample]:
    result = []
    with open("checkout/result.csv") as fin:
        reader = csv.reader(fin)
        for _, succ, _, samples in reader:
            succ = succ == "True"
            if not succ:
                continue
            samples = json.loads(samples)
            for sample in samples:
                env_file = sample[0]
                token_file = sample[1]
                result.append(RunningSample(env_file, token_file))
    return result
