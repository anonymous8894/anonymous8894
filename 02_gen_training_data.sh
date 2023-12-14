
# C
pushd utils/tracer/data/dataset/singleL
unzip singleL_Test.zip
unzip singleL_Train+Valid.zip
popd
pushd neural_models/c
python process_all.py
python c_generate_train.py
popd

# Middleweight Java
pushd neural_models/middleweight-java/data
python gen_env.py
python mutant.py
popd
python scripts/mj-test-training-large.py
pushd neural_models/middleweight-java
python extract-data-from-javagrammar.py

