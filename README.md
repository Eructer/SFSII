# SFSII
Solar Filament Segmentation Implementing IBNN


## Notes on IBNN Code

- Line `2499` : Base function for `ibnn_internal` and `INRF`, this is where we see the IBNN bias added to the standard ANN
- Line `1733` : Per Pixel computation

### What we can re-use

- `modified_rf.nn_layers` : Functions (`IBNNLiteLayer`, `IBNNLayer`, `SMLayer`)

### What to replace for ViT or other segmentation

- `obtain_classification_dataset_loaders` : Datasets are currently hardcoded
  - Write a normal PyTorch `Dataset` and `DataLoader`
- `ClassifierBaseModel` and subclasses
- `classifier_training` : The training is build for cross-entropy
  - We need our own loop using either Dice or CE Loss and IoU
