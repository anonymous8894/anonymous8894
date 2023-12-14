import csv
import dataclasses
import itertools
import json
import multiprocessing
import os
import re
import signal
import subprocess
import typing

import javalang
import pandas as pd
import tqdm
from nltk.metrics.distance import edit_distance


@dataclasses.dataclass
class TestInstance:
    input_env: str
    input_block: str
    output_block: str
    output_log: str


GLOBAL_SIGINT = False


def _subprocess_run(args):
    global GLOBAL_SIGINT
    if GLOBAL_SIGINT:
        return

    cmdline, timeout, path = args
    p = subprocess.Popen(
        cmdline,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    guessed_result = False
    try:
        txt = ""
        try:
            stdout, stderr = p.communicate(None, timeout)
            stdout = stdout.decode("utf-8", errors="ignore").splitlines()
            stderr = stderr.decode("utf-8", errors="ignore").splitlines()
            for line in itertools.chain(stdout, stderr):
                if "RESULT" in line and "length:" in line and "length:-" not in line:
                    guessed_result = True
                txt = txt + "\n" + line.strip()
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            info = f"{path} time out."
            print(info)
            txt = txt + "\n" + info
        p.wait()
        with open(path, "wt") as fout:
            fout.write(txt)
    except KeyboardInterrupt:
        GLOBAL_SIGINT = True
        return
    finally:
        if p.poll() is None:
            print("Killing children")
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait()
    return guessed_result, path


class _DummyPool:
    def imap_unordered(self, func, iterable):
        for i in iterable:
            yield func(i)

    def join(self):
        pass

    def close(self):
        pass


class Runner:
    def __init__(
        self,
        cmd_line_generator: typing.Callable[[TestInstance], list[str]],
        timeout=600,
        pool_size=None,
        show_fail=False,
    ):
        self.cmd_line_generator = cmd_line_generator
        self.timeout = timeout
        self.pool_size = pool_size
        self.show_fail = show_fail

    def run(self, instances: list[TestInstance]) -> None:
        global GLOBAL_SIGINT

        if isinstance(self.pool_size, int) and self.pool_size < 0:
            pool = _DummyPool()
        else:
            pool = multiprocessing.Pool(self.pool_size)
        instances = [
            (self.cmd_line_generator(instance), self.timeout, instance.output_log)
            for instance in instances
        ]
        try:
            succ = 0
            finished = 0
            t = tqdm.tqdm(
                pool.imap_unordered(_subprocess_run, instances), total=len(instances)
            )
            for result, path in t:
                finished += 1
                if result:
                    succ += 1
                else:
                    if self.show_fail:
                        print("FAIL", path)
                t.set_description(f"SUCC: {succ}/{finished}")
            pool.close()
        except KeyboardInterrupt:
            GLOBAL_SIGINT = True
            pool.close()
            pool.join()
            raise


class CmdLineGenerator:
    def __init__(
        self,
        lang: str,
        use_symbolic_executable=False,
        symbolic_executable: str | None = None,
        use_kissat=False,
        max_len=8,
        memory_limit_gb=16,
        executable: str | None = None,
        use_neural=False,
        neural_name=None,
    ):
        self.memory_limit = str(int(memory_limit_gb * 1024 * 1024 * 1024))
        self.max_len = str(max_len)
        self.lang = lang
        if executable is None:
            executable = os.path.join(
                os.path.split(os.path.split(os.path.abspath(__file__))[0])[0],
                "ordinal-fix-modified",
                "target",
                "release",
                "fixing-rs-main",
            )
        self.executable = executable
        self.symbolic_cmdline = []
        self.use_neural = use_neural
        self.neural_name = neural_name

    def __call__(self, instance: TestInstance) -> list[str]:
        if self.use_neural:
            neural_cmdline = [
                "--weights",
                instance.input_block
                + (self.neural_name if self.neural_name is not None else "_neural"),
            ]
        else:
            neural_cmdline = []
        return [
            self.executable,
            "--memory-limit",
            self.memory_limit,
            "fix",
            "--lang",
            self.lang,
            "--max-len",
            self.max_len,
            "--max-new-id",
            self.max_len,
            *self.symbolic_cmdline,
            "single",
            "--input",
            instance.input_block,
            "--env",
            instance.input_env,
            "--output",
            instance.output_block,
            *neural_cmdline,
        ]


AddColGenerator = typing.Callable[[None | TestInstance], list]


class Stater:
    def __init__(self, add_col_generator: AddColGenerator | None = None):
        self.add_col_generator = add_col_generator

    def _stat(self, instances: list[TestInstance], writer):
        if self.add_col_generator is not None:
            add_cols = self.add_col_generator(None)
        else:
            add_cols = []
        writer(
            (
                "block",
                "total_time",
                "dist",
                "err",
                *add_cols,
            )
        )
        for instance in tqdm.tqdm(instances, desc="STAT "):
            err = ""
            total_time = -1
            dist = -2
            info_file = instance.output_log
            if os.path.exists(info_file):
                with open(info_file) as fin:
                    for line in fin:
                        line = line.strip()
                        if line.startswith("---RESULT---"):
                            line = line[13:]
                            line = line.split(",")
                            line = [x.split(":") for x in line]
                            line = [(x[0], x[1]) for x in line]
                            line = dict(line)
                            if "error" in line:
                                err = line["error"]
                            else:
                                total_time = (
                                    float(line["time_load"])
                                    + float(line["time_build"])
                                    + float(line["time_find"])
                                )
                                dist = int(line["length"])
            if self.add_col_generator is not None:
                add_cols = self.add_col_generator(instance)
            writer((instance.input_block, total_time, dist, err, *add_cols))

    def stat(
        self, instances: list[TestInstance], output_file: str | None
    ) -> None | pd.DataFrame:
        if output_file is not None:
            with open(output_file, "wt", newline="") as fout:
                writer = csv.writer(fout)
                self._stat(instances, lambda x: writer.writerow(x))
        else:
            results = []
            self._stat(instances, lambda x: results.append(x))
            return pd.DataFrame(results[1:], columns=results[0])


class MJMutAddColGenerator:
    PATH_RE = re.compile(r".*/m_([a-z]*)_([0-9]*)/([0-9a-z]*)/?.*")
    LINE_BREAKERS = set(("{", "}", ";"))

    def __init__(self):
        pass

    def __call__(self, instance: None | TestInstance) -> list:
        if instance is None:
            return ["cls_mod", "mut_mod", "num_tokens", "num_lines", "edit_distance"]
        m = MJMutAddColGenerator.PATH_RE.match(instance.input_block)
        if m:
            cls_mod = m.group(1)
            mut_mod = int(m.group(2))
        else:
            cls_mod = "unknown"
            mut_mod = -1
        with open(instance.input_block) as fin:
            tokens = [t.value for t in javalang.tokenizer.tokenize(fin.read())]
        num_tokens = len(tokens)
        num_lines = sum(1 for t in tokens if t in MJMutAddColGenerator.LINE_BREAKERS)

        tokens_map = list(set(tokens))
        tokens_map_inv = {t: i for i, t in enumerate(tokens_map)}

        if os.path.exists(instance.output_block):
            with open(instance.output_block) as fin:
                output_tokens = json.load(fin)
            if len(output_tokens) > 0:
                output_tokens = [x["value"] or x["name"] for x in output_tokens[0]]
                tokens = [tokens_map_inv.get(t, -1) for t in tokens]
                output_tokens = [tokens_map_inv.get(t, -1) for t in output_tokens]
                e = edit_distance(tokens, output_tokens)
            else:
                e = -1
        else:
            e = -1

        return [cls_mod, mut_mod, num_tokens, num_lines, e]
