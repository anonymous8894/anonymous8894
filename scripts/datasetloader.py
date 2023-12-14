import runner
import random
import os


def to_mj_dataset(folders, test_name) -> list[runner.TestInstance]:
    results = []
    for f in folders:
        results.append(
            runner.TestInstance(
                f"{f}/env",
                f"{f}/block",
                f"{f}/output_{test_name}",
                f"{f}/runinfo_{test_name}",
            )
        )
    return results


def load_mj_test(test_name, min_mut=1, max_mut=8) -> list[runner.TestInstance]:
    folders = []
    r = random.Random(0)
    for mutant_type in ["p", "pi", "i", "a"]:
        for mutant_count in range(1, 9):
            d = f"dataset/mj/m_{mutant_type}_{mutant_count}"
            folders_new = os.listdir(d)
            folders_new = sorted(folders_new)
            folders_new = [os.path.join(d, folder) for folder in folders_new]
            r.shuffle(folders_new)
            folders_new = folders_new[:100]
            folders_new.sort()
            if mutant_count >= min_mut and mutant_count <= max_mut:
                folders.extend(folders_new)
    return to_mj_dataset(folders, test_name)



def mj_to_check(instances: list[runner.TestInstance]) -> list[runner.TestInstance]:
    results = []
    for inst in instances:
        if os.path.exists(inst.output_block):
            results.append(
                runner.TestInstance(
                    inst.input_env,
                    inst.output_block,
                    inst.output_block + "_check",
                    inst.output_log + "_check",
                )
            )
    return results
