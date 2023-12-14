import c_model
import pickle


def main():
    test_limit = 1000
    trainer = c_model.ModifyingModelTrainer(
        test_length_limit=test_limit, test_shuffle=True, test_shuffle_seed=0
    )
    trainer.load_model()
    outputs_results, insertions_outputs = trainer.test()
    test_dataset = trainer.dataset_test.dataset._data[:test_limit]
    test_dataset = [(None, x.inputs, x.outputs, x.insertions) for x in test_dataset]
    with open("c-test.pkl", "wb") as fout:
        pickle.dump((test_dataset, outputs_results, insertions_outputs), fout)


if __name__ == "__main__":
    main()
