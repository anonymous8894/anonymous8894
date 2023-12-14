import process_all
import sys


def main():
    files = sys.argv[1:]
    for f in files:
        with open(f) as fin:
            text = fin.read()
        seg = process_all.CodeSegment(
            file=None,
            line=None,
            flags=None,
            begin_line=None,
            is_original=None,
            code=text,
            tokens=None,
            truncated_tokens=None,
            braces=None,
            include_line=None,
        )
        process_all.tokenize_segment(seg, set())
        tokens = seg.tokens
        with open(f + ".tokens", "w") as fout:
            fout.write(
                "\n".join(
                    f"{t.type}\t{t.name}\t{t.value}" for t in tokens
                )
            )

if __name__ == "__main__":
    main()
