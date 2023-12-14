import csv
import json
import os
import collections


def load_try_compile():
    with open("checkout/try_compile.txt") as fin:
        return set(line.strip() for line in fin if line.strip())



def main():
    count_succ = 0
    count_all = 0
    failed_to_compile = load_try_compile()
    with open("checkout/result.csv") as fin:
        reader = csv.reader(fin)
        reason_coll = collections.Counter()
        for _, succ, reason, samples in reader:
            count_all += 1
            if reason == "all_tokens_compile":
                count_succ += 1
                reason_coll[reason] += 1
                continue
            succ = succ == "True"
            if not succ:
                reason_coll[reason] += 1
                continue
            samples = json.loads(samples)
            for sample in samples:
                token_file = sample[1]
                run_info_file = token_file + "run_info.txt"
                fix_file = token_file + ".fix.ass.c"
                if (
                    fix_file in failed_to_compile
                    or not os.path.exists(fix_file)
                    or not os.path.exists(run_info_file)
                ):
                    succ = False
                    break
                guess_succ = False
                with open(run_info_file) as fin:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("---RESULT---"):
                            line = line[13:]
                            line = line.split(",")
                            line = [x.split(":") for x in line]
                            line = [(x[0], x[1]) for x in line]
                            line = dict(line)
                            if "length" in line and line["length"] != "-1":
                                guess_succ = True
                                break
                if not guess_succ:
                    succ = False
                    break
            if succ:
                count_succ += 1
    print(f"count_succ: {count_succ}")
    print(f"count_all: {count_all}")
    print(f"ratio: {count_succ / count_all}")
    for reason, count in reason_coll.most_common():
        print(f"{reason}: {count}")


if __name__ == "__main__":
    main()
