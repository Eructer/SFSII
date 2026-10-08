# SFSII
Solar Filament Segmentation Implementing IBNN

# Notes on IBNN Code

## What we can re-use

- `modified_rf.nn_layers` : Classes (`IBNNLiteLayer`, `IBNNLayer`, `SMLayer`)

We can just build on top of them and shouldn't need to modify the code

But we do need to make sure we have all the subclass dependencies when using in project

### `IBNNLayer`

Layer implemetating the Implicitly-Biased Neural Network layer

Extends
- `IBNNInternalLayer`

Parameters come from `IBNNInternalLayer`

### `IBNNLiteLayer`

Layer implementing a simplified version of the IBNN layer

Extends
- `INRFFamilyLayer`

Parameters
- `in_channels` : `int`, number of input channels
- `out_channels` : `int`, number of output channels
- `phi_activation` : `str` or `list[str]`, phi actication function?
- `m_kernel_size` : `tuple[int]` or `tuple[float]` or `list[int]` or `list[float]` or `int` or `float`
- `m_padding` : `str` or optional
- `m_padding_mode` : `str` or optional
- `m_groups` : `int` or optional
- `m_initialization` : `str` or `dict[str]` or `torch.Tensor` or `int` or `float` or optional
- `m_trainable` : `bool` or optional
- `in_size` : `tuple[int]` or optional
- `sigma_actication` : `str` or `list[str]`
- `sigma_x_compress` : `int` or `float` or `list[int or float]` or `tuple[int or float]`
- `sigma_y_stretch` : `int` or `float` or `list[int or float]` or `tuple[int or float]`
- `sigma_x_offset` : `int` or `float` or `list[int or float]` or `tuple[int or float]`
- `sigma_y_offset` : `int` or `float` or `list[int or float]` or `tuple[int or float]`
- `sigma_x_compress_trainable` : `bool` or optional
- `sigma_y_compress_trainable` : `bool` or optional
- `lambda_type` : `str` or optional
- `initial_lambda` : `int` or `float` or `torch.Tensor`
- `lambda_trainable` : `bool` or optional
- `w_kernel_size` : `tuple[int]` or `tuple[float]` or `list[int]` or `list[float]` or `int` or `float`
- `w_padding_mode` : `str` or optional
- `w_groups` : `int` or optional
- `w_intitialization` : `str` or `dict[str]` or `torch.Tensor` or `int` or `float` or optional
- `w_trainable` : `bool` or optional


### `SMLayer`

Layer implementing the Standard Model neural network layer

Extends
- `ModifiedRFLayer`
- `nn.Module`

Parameters
- `in_channels` : `int`, input channels of image
- `out_channels` : `int`, output channels of image
- `phi_activation` : `str` or `list[str]`
- `m_kernel_size` : `tuple[int]`, size of the kernel convolving the input image
- `m_padding` : `str` or optional
- `m_padding_mode` : `str` or optional
- `m_groups` : `int` or optional
- `m_initialization` : `str` or `dict[str]`, Initialization mode for the filter mask
- `m_trainable` : `bool` or optional, indicates whether the corresponding element of theta performs gradient calculations or remains with its initial value
- `b_type` : `str` or optional, defines whether the same bias, although independent per channel, is shared for the whole image
- `initial_b` : `int` or `float` or `torch.Tensor`
- `b_trainable` : `bool` or optional, indicates whether the corresponding element of theta performs gradient calculations or remains with its initial value
- `in_size` : `tuple[int]` or optional, 2D tuple indicating the size of the input image to the layer

### What to replace for ViT or other segmentation

- `obtain_classification_dataset_loaders` : Datasets are currently hardcoded
  - Write a normal PyTorch `Dataset` and `DataLoader`
- `ClassifierBaseModel` and subclasses
- `classifier_training` : The training is build for cross-entropy
  - We need our own loop using either Dice or CE Loss and IoU
