import os
import pickle

import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, BertModel

import c_hyper_parameters as hyper_parameters
import c_embedding


class ModifyingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._bert_config = BertConfig(
            vocab_size=c_embedding.num_tokens,
            pad_token_id=c_embedding.v_pad,
        )
        self.bert = BertModel(self._bert_config, add_pooling_layer=False)
        self.ff = torch.nn.Linear(768, 3072)
        self.output = torch.nn.Linear(3072, c_embedding.num_tokens)
        self.insertions = torch.nn.Linear(
            3072, c_embedding.num_tokens * hyper_parameters.num_insertions
        )

    def forward(self, x):
        batch_size, sequence_length = x.shape
        attention_mask = (x != c_embedding.v_pad).long()
        x = self.bert(x, attention_mask=attention_mask).last_hidden_state
        # x = self.bert(x).last_hidden_state
        x = self.ff(x)
        x = torch.nn.functional.relu(x)
        outputs = self.output(x)
        insertions = self.insertions(x).view(
            batch_size,
            sequence_length,
            hyper_parameters.num_insertions,
            c_embedding.num_tokens,
        )

        return outputs, insertions


def modifying_model_loss(outputs, insertions, target_outputs, target_insertions):
    padding = target_outputs == c_embedding.v_pad
    batch_size, sequence_length = target_outputs.shape
    non_zero_nums = torch.sum(~padding, dim=1)

    outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
    insertions = torch.nn.functional.log_softmax(insertions, dim=-1)
    loss1 = torch.nn.functional.nll_loss(
        outputs.view(batch_size * sequence_length, -1),
        target_outputs.view(-1),
        reduction="none",
    ).view(batch_size, sequence_length)
    loss1 = loss1.masked_fill(padding, 0.0).sum(dim=1) / non_zero_nums
    loss1 = loss1.mean()
    loss2 = torch.nn.functional.nll_loss(
        insertions.view(
            batch_size * sequence_length * hyper_parameters.num_insertions, -1
        ),
        target_insertions.view(-1),
        reduction="none",
    ).view(batch_size, sequence_length, hyper_parameters.num_insertions)
    loss2 = (
        loss2.masked_fill(padding.unsqueeze(-1), 0.0).sum(dim=1).sum(dim=1)
        / hyper_parameters.num_insertions
        / non_zero_nums
    )
    loss2 = loss2.mean()
    return (loss1 + loss2) / 2


class ModifyingModelDataset:
    def __init__(self, file_name):
        with open(file_name, "rb") as f:
            self._data = pickle.load(f)

    def __getitem__(self, idx):
        # _, inputs, outputs, insertions = self._data[idx]
        data = self._data[idx]
        inputs, outputs, insertions = (
            data.inputs,
            data.outputs,
            data.insertions,
        )
        inputs = torch.tensor(inputs, dtype=torch.long)
        outputs = torch.tensor(outputs, dtype=torch.long)
        insertions = torch.tensor(insertions, dtype=torch.long)
        return inputs, outputs, insertions

    def __len__(self):
        return len(self._data)


class ModifyingModelTrainer:
    def __init__(self, load_dataset=True, load_writer=True):
        if not torch.cuda.is_available():
            raise Exception("No GPU found")
        self.device = torch.device("cuda:0")
        # self.device = torch.device("cpu")
        self.model = ModifyingModel().to(self.device)

        if load_dataset:
            dataset_training = ModifyingModelDataset("checkout/c-train-pretraining.pkl")
            dataset_validation = ModifyingModelDataset(
                "checkout/c-valid-pretraining.pkl"
            )
            dataset_test = ModifyingModelDataset("checkout/c-test-pretraining.pkl")

            self.dataset_training = torch.utils.data.DataLoader(
                dataset_training, batch_size=hyper_parameters.batch_size, shuffle=True
            )
            self.dataset_validation = torch.utils.data.DataLoader(
                dataset_validation,
                batch_size=hyper_parameters.batch_size,
                shuffle=False,
            )
            self.dataset_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=hyper_parameters.batch_size, shuffle=False
            )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=hyper_parameters.learning_rate
        )
        self.lowest_valid_loss = float("inf")
        self.number_epoches = 0
        self.number_steps = 0
        if load_writer:
            self.writer = SummaryWriter()
        else:
            self.writer = None

    def save_model(self):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lowest_valid_loss": self.lowest_valid_loss,
                "number_epoches": self.number_epoches,
                "number_steps": self.number_steps,
            },
            self.get_model_file_name(),
        )

    def get_model_file_name(self):
        return os.path.join(os.path.dirname(__file__), "c_model_pretraining.pt")

    def load_model(self):
        if not os.path.exists(self.get_model_file_name()):
            print("model not found")
            return
        checkpoint = torch.load(self.get_model_file_name())
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lowest_valid_loss = checkpoint["lowest_valid_loss"]
        self.number_epoches = checkpoint["number_epoches"]
        self.number_steps = checkpoint["number_steps"]
        print("model loaded")

    def train_epoch(self):
        self.model.train()
        t = tqdm.tqdm(self.dataset_training)
        for inputs, outputs, insertions in t:
            inputs = inputs.to(self.device)
            outputs = outputs.to(self.device)
            insertions = insertions.to(self.device)
            outputs_p, insertions_p = self.model(inputs)
            loss = modifying_model_loss(outputs_p, insertions_p, outputs, insertions)
            if loss == float("inf"):
                print(self.number_steps)
                print(inputs == c_embedding.v_pad)
                raise Exception("inf occured")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            t.set_description(
                f"E{self.number_epoches:08d} S{self.number_steps:08d} L{loss.item():09.06f}"
            )
            self.number_steps += 1
            if self.writer is not None:
                self.writer.add_scalar("C-p/Loss/train", loss.item(), self.number_steps)
        self.number_epoches += 1

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            num_valid = 0
            for inputs, outputs, insertions in tqdm.tqdm(
                self.dataset_validation, desc="VALIDATION"
            ):
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                insertions = insertions.to(self.device)
                outputs_p, insertions_p = self.model(inputs)
                loss = modifying_model_loss(
                    outputs_p, insertions_p, outputs, insertions
                )
                valid_loss += loss.item()
                num_valid += 1
            valid_loss /= num_valid
            if valid_loss < self.lowest_valid_loss:
                self.lowest_valid_loss = valid_loss
                self.save_model()
            if self.writer is not None:
                self.writer.add_scalar("C-p/Loss/valid", valid_loss, self.number_steps)
            print(f"Epoch {self.number_epoches} validation loss {valid_loss}")

    def run_instance(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.long)
            should_squeeze = False
            if len(inputs.shape) == 1:
                inputs = inputs.unsqueeze(0)
                should_squeeze = True
            inputs = inputs.to(self.device)
            outputs_p, insertions_p = self.model(inputs)
            outputs_p_softmax = torch.log_softmax(outputs_p, dim=-1)
            insertions_p_softmax = torch.log_softmax(insertions_p, dim=-1)
            if should_squeeze:
                outputs_p_softmax = outputs_p_softmax.squeeze(0)
                insertions_p_softmax = insertions_p_softmax.squeeze(0)
            return outputs_p_softmax.cpu().numpy(), insertions_p_softmax.cpu().numpy()

    def test(self, need_result=True):
        self.model.eval()
        with torch.no_grad():
            output_corr, output_total, output_copy = 0, 0, 0
            insertion_corr, insertion_total, insertion_empty = (
                0,
                0,
                0,
            )

            if need_result:
                outputs_results, insertions_outputs = [], []

            for inputs, outputs, insertions in tqdm.tqdm(
                self.dataset_test, desc="TEST"
            ):
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                insertions = insertions.to(self.device)
                outputs_p, insertions_p = self.model(inputs)
                outputs_p_softmax = torch.softmax(outputs_p, dim=-1)
                insertions_p_softmax = torch.softmax(insertions_p, dim=-1)
                if need_result:
                    outputs_results.append(outputs_p_softmax.cpu().numpy())
                    insertions_outputs.append(insertions_p_softmax.cpu().numpy())
                outputs_p = outputs_p.argmax(dim=-1)
                insertions_p = insertions_p.argmax(dim=-1)
                padding_mask = inputs == c_embedding.v_pad
                output_corr += (
                    (outputs_p == outputs).masked_fill(padding_mask, False).sum().item()
                )
                output_total += (~padding_mask).sum().item()
                output_copy += (
                    (outputs_p == c_embedding.v_copy)
                    .masked_fill(padding_mask, False)
                    .sum()
                    .item()
                )
                insertion_corr += (
                    (insertions_p == insertions)
                    .masked_fill(padding_mask.unsqueeze(-1), False)
                    .sum()
                    .item()
                )
                insertion_total += (
                    ~padding_mask
                ).sum().item() * hyper_parameters.num_insertions
                insertion_empty += (
                    (insertions_p == c_embedding.v_remove)
                    .masked_fill(padding_mask.unsqueeze(-1), False)
                    .sum()
                    .item()
                )
            print(f"Epoch {self.number_epoches}")
            print(f"output acc:{output_corr/output_total*100:07.04f}%")
            print(f"output copy:{output_copy/output_total*100:07.04f}%")
            print(f"insertion acc:{insertion_corr/insertion_total*100:07.04f}%")
            print(f"insertion empty:{insertion_empty/insertion_total*100:07.04f}%")
            if self.writer is not None:
                self.writer.add_scalar(
                    "C-p/Output_Acc", output_corr / output_total, self.number_steps
                )
                self.writer.add_scalar(
                    "C-p/Insertion_Acc",
                    insertion_corr / insertion_total,
                    self.number_steps,
                )

        if need_result:
            outputs_results = np.concatenate(outputs_results)
            insertions_outputs = np.concatenate(insertions_outputs)
            return outputs_results, insertions_outputs

    def run_train(self):
        self.load_model()
        for epoch in range(100):
            self.train_epoch()
            self.validate()
            self.test(False)

    def run_test_output(self):
        self.load_model()
        test_result = self.test()
        outputs_results, insertions_outputs = test_result
        outputs_results = outputs_results[:1000]
        insertions_outputs = insertions_outputs[:1000]
        test_result = outputs_results, insertions_outputs

        with open("test_result-c-pretraining.pkl", "wb") as f:
            pickle.dump(test_result, f)


def main():
    trainer = ModifyingModelTrainer()
    trainer.run_train()


if __name__ == "__main__":
    main()
