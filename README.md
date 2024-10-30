# Training isiZulu Language Model on Surface Morphological Segmentations

## Author: Temiloluwa Aina

### Project Supervisor: Dr. Jan Buys

This project extends the work done by Moeng et al. 2021 to train a language model on surface segmentations using the CRF segmentation model developed in their research. (https://github.com/DarkPr0digy/MORPH_SEGMENT)

## Morphological Segmentation

Morphological segmentation involves breaking up words into their composite morphemes (e.g., the isiZulu word “ngezinkonzo” could be broken up into morphemes “nge-“, “-zin” and “-konzo”). These segmentations allow language models to infer the meaning of unseen words if the morphemes that the unseen word is composed of have been seen during training.

### This Project

This project contains the code necessary to train a CRF segmentation model and a ULM that is trained on surface segments.

To train CRF and ULM model run the preprocessing.py file.

The language model was trained using nanoGPT (https://github.com/karpathy/nanoGPT). Run the prepare.py file (changing the filenames where necessary) to obtain the "train.bin" and "valid.bin" to necessary for the language model training.

The models folder contains the CRF model and both version of the Unigram models trained in our research our results can be verified.

isiZuluSurfaceModel.sav is the CRF model.

morph_final.model is the Unigram model trained on surface segments.

normal_ulm.model is the Unigram model trained on the unprocessed text.

The data used for language model training can be found here: https://huggingface.co/datasets/castorini/wura

The outputs of the model training can be found in the results folder.