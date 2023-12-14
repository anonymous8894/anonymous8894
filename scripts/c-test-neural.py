import c_datasetloader
import runner

if __name__ == "__main__":
    items = c_datasetloader.get_files("test_neural")
    cmdline_generator = runner.CmdLineGenerator(
        lang="c", use_symbolic_executable=False, max_len=100, use_neural=True
    )
    r = runner.Runner(cmdline_generator, pool_size=8)
    r.run(items)
    stater = runner.Stater()
    stater.stat(items, "results/c-test-neural.csv")
