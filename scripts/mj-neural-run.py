import datasetloader
import sys
import tqdm
import traceback

if __name__ == "__main__":
    sys.path.append("./neural_models/middleweight-java")
    import process_mj_file
    import model

    trainer = model.ModifyingModelTrainer(load_dataset=False, load_writer=False)
    trainer.load_model()
    items = datasetloader.load_mj_test("test-neural")
    for item in tqdm.tqdm(items):
        try:
            process_mj_file.process_mj_file(
                item.input_block,
                item.input_env,
                item.input_block + "_neural",
                100,
                trainer,
                10,
            )
        except Exception:
            traceback.print_exc()
