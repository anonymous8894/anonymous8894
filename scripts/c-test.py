import c_datasetloader
import runner

if __name__ == "__main__":
    items = c_datasetloader.get_files("test")
    cmdline_generator = runner.CmdLineGenerator(
        lang="c", use_symbolic_executable=False, max_len=20
    )
    for item in items:
        item.output_block = f"{item.input_block}.fix-dedup.c"
        item.output_log = f"{item.input_block}run_info.txt"
    r = runner.Runner(cmdline_generator, pool_size=8)
    r.run(items)
    stater = runner.Stater()
    stater.stat(items, "results/c-test.csv")
