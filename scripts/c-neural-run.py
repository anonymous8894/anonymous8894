import sys

sys.path.append("./neural_models/c/")

import traceback

import c_datasetloader
import c_model
import c_process_file
import tqdm

if __name__ == "__main__":
    trainer = c_model.ModifyingModelTrainer(load_dataset=False, load_writer=False)
    trainer.load_model()
    items = c_datasetloader.get_files("test-neural")
    for item in tqdm.tqdm(items):
        try:
            tokens = c_process_file.process_file(
                item.input_block,
                item.input_env,
                trainer,
                100,
                item.input_block + "_neural",
                10,
            )
        except Exception:
            traceback.print_exc()
            raise
