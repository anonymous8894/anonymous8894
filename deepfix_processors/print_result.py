ignore_errors = [
    # "assignment of read-only variable",
    # "error: array size missing in",
    # "‘LITERAL_STRING’ undeclared",
    # "error: ‘LITERAL_FLOAT’ undeclared",
    # "invalid initializer"
    # "is not a function or function pointer"
]

count = 0
with open("checkout/try_compile.txt") as fin:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        result_file = line + ".result"
        errors = []
        with open(result_file) as fresult:
            for l in fresult:
                l = l.strip()
                if not l:
                    continue
                if "error:" in l:
                    ignored = False
                    for err in ignore_errors:
                        if err in l:
                            ignored = True
                            break
                    if not ignored:
                        errors.append(l)
        if len(errors):
            count += 1
            print(line)
            print(result_file)
            for err in errors:
                print(err)

print(count)