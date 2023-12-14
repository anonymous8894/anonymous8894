import os

with open("checkout/try_compile.txt") as fin:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        assert line.endswith(".fix.ass.c")
        line = line[:-10] + ".fix.c"
        os.unlink(line)
