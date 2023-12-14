import datasetloader
import runner

if __name__ == "__main__":
    items = datasetloader.load_mj_test("test")
    cmdline_generator = runner.CmdLineGenerator(
        lang="mj", use_symbolic_executable=False, max_len=10
    )
    r = runner.Runner(cmdline_generator, pool_size=8)
    r.run(items)
    stater = runner.Stater(runner.MJMutAddColGenerator())
    stater.stat(items, "results/mj-test.csv")
