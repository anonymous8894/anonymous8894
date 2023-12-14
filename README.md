
# Code repository for NGSeFix

# Requirement

- Rust (nightly build) (`rustup toolchain install nightly`)
- Python Environment (`conda env create -n torch21 -f torch21.yaml`)

# Scripts

To run the experiments, simply execute 01_xxxx.sh, 02_xxxx.sh, and so on, in sequential order.

- `01_build.sh` builds the adapted OrdinalFix algorithm.
- `02_gen_training_data.sh` generates training data for neural models.
- `03_train_mj.sh` trains neural model for Middleweight Java.
- `04_train_c.sh` trains neural model for Middleweight C.
- `05_deepfix_prepare.sh` prepares evaluation data for Middleweight Java.
- `06_middleweightjava_prepare.sh` prepares evaluation data for C.
- `07_mj_original.sh` evaluates the performance of the original OrdinalFix on Middleweight Java.
- `08_mj_neural_prediciton.sh` evaluates the performance of NGSeFix on Middleweight Java.
- `09_c_original.sh` evaluates the performance of the original OrdinalFix on C.
- `10_c_neural_prediction.sh` evaluates the performance of NGSeFix on C.

The results will be shown in folder `results`.

# Anonymity

This repository includes code from third-party sources, including OrdinalFix, pycparser (utils/fake_libc_include), TRACER (utils/tracer). The original authors' information is included within, and does not pertain to the authorship of this document.
