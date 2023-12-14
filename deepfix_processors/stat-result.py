import csv
import os

import processed_loader
import stat_final


def get_num_tokens(path):
    num_tokens = 0
    with open(path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            num_tokens += 1
    return num_tokens


LINE_BREAKERS = set(("{", "}", ";"))


def main():
    samples = processed_loader.get_running_samples()
    failed_to_compile = stat_final.load_try_compile()
    succ = 0
    memout = 0
    timeout = 0
    compf = 0
    total = 0
    with open("checkout/stat.csv", "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            (
                "token_file",
                "num_tokens",
                "total_time",
                "dist",
                "num_tokens",
                "num_lines",
            )
        )
        for sample in samples:
            ass = sample.token_file + ".fix.ass.c"
            output_file = sample.token_file + ".fix.c"
            info_file = sample.token_file + "run_info.txt"
            num_tokens = get_num_tokens(sample.token_file)
            total_time = -1
            dist = -1
            with open(sample.token_file) as fin:
                tokens = fin.readlines()
            num_tokens = len(tokens)
            num_lines = 0
            for line in tokens:
                line = line.strip()
                if not line:
                    continue
                if line.split("\t")[2] in LINE_BREAKERS:
                    num_lines += 1

            # if ass in failed_to_compile:
            #     print("FAIL TO COMPILE")


            ismemout = False
            istimeout = False

            if ass not in failed_to_compile and os.path.exists(info_file):
                with open(info_file) as fin:
                    for line in fin:
                        line = line.strip()
                        line_o = line
                        if line.startswith("---RESULT---"):
                            line = line[13:]
                            line = line.split(",")
                            line = [x.split(":") for x in line]
                            line = [(x[0], x[1]) for x in line]
                            line = dict(line)
                            total_time = (
                                float(line["time_load"])
                                + float(line["time_build"])
                                + float(line["time_find"])
                            )
                            dist = int(line["length"])
                        line = line_o
                        if 'memory allocation of ' in line and ' bytes failed' in line:
                            ismemout = True
                        if 'TIMEOUT' in line:
                            istimeout = True
            if ismemout:
                memout += 1
            if istimeout:
                timeout += 1
            if ass in failed_to_compile:
                compf += 1
            writer.writerow(
                (sample.token_file, num_tokens, total_time, dist, num_tokens, num_lines)
            )

            succ += int(dist != -1)
            total += 1

    print("succ", succ)
    print("memout", memout)
    print("timeout", timeout)
    print("compf", compf)
    print("total", total)

    for a in failed_to_compile:
        print(a)



if __name__ == "__main__":
    main()
