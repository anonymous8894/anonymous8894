import os
import re
import processed_loader

MEM_FAIL_LINE_BEGIN = "memory allocation of "
MEM_FAIL_LINE_END = " bytes failed"

MEM_ALLOC_BIG = re.compile(r".*alloc: ([0-9]+) bytes\..*")


def main():
    max_amount = -1
    processed = processed_loader.get_running_samples()
    for sample in processed:
        fix_file = sample.token_file + ".fix.ass.c"
        run_info_file = sample.token_file + "run_info.txt"
        if not os.path.exists(fix_file):
            with open(run_info_file) as fin:
                printed = False
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    if not printed:
                        m = MEM_ALLOC_BIG.match(line)
                        if m:
                            print(run_info_file)
                            printed = True
                    if line.startswith(MEM_FAIL_LINE_BEGIN) and line.endswith(
                        MEM_FAIL_LINE_END
                    ):
                        amount = int(
                            line[len(MEM_FAIL_LINE_BEGIN) : -len(MEM_FAIL_LINE_END)]
                        )
                        max_amount = max(max_amount, amount)
                        # print(amount)
    print("max_amount: ")
    print(max_amount)


if __name__ == "__main__":
    main()
