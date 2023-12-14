import csv
import json
import random
import runner
import typing

BASE_FOLDER = "deepfix_processors/"


def get_files(output_file_name: str) -> typing.List[runner.TestInstance]:
    files = []
    with open(BASE_FOLDER + "checkout/result.csv") as fin:
        reader = csv.reader(fin)
        for _, succ, _, samples in reader:
            succ = succ == "True"
            if not succ:
                continue
            samples = json.loads(samples)
            for sample in samples:
                env_file = BASE_FOLDER + sample[0] + ".p"
                token_file = BASE_FOLDER + sample[1]
                output_file = BASE_FOLDER + sample[1] + "_output_" + output_file_name
                runinfo_file = BASE_FOLDER + sample[1] + "_runinfo_" + output_file_name
                files.append(
                    runner.TestInstance(env_file, token_file, output_file, runinfo_file)
                )
    r = random.Random(0)
    r.shuffle(files)
    return files
