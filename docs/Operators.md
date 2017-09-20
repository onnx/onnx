## Operator Schemas
* **Abs**

  Absolute takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the absolute is, y = abs(x), is applied to
  the tensor elementwise.
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output tensor</dd>
    </dl>


* **Add**

  Performs element-wise binary addition (with limited broadcast support).
  
  If necessary the right-hand-side argument will be broadcasted to match the
  shape of left-hand-side argument. When broadcasting is specified, the second
  tensor can either be of size 1 (a scalar value), or having its shape as a
  contiguous subset of the first tensor's shape. The starting of the mutually
  equal shape is specified by the argument "axis", and if it is not set, suffix
  matching is assumed. 1-dim expansion doesn't work yet.
  
  For example, the following tensor shapes are supported (with broadcast=1):
  
    shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
    shape(A) = (2, 3, 4, 5), shape(B) = (5,)
    shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
    shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
    shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
  
  Attribute `broadcast=1` needs to be passed to enable broadcasting.
  * **attribute**:
    <dl>
      <dt>axis</dt>
      <dd>If set, defines the broadcast dimensions. See doc for details.</dd>
      <dt>broadcast</dt>
      <dd>Pass 1 to enable broadcasting</dd>
    </dl>
  * **input**:
    <dl>
      <dt>A</dt>
      <dd>First operand, should share the type with the second operand.</dd>
      <dt>B</dt>
      <dd>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>C</dt>
      <dd>Result, has same dimensions and type as A</dd>
    </dl>


* **ArgMax**

  Computes the indices of the max elements of the input tensor's element along the 
  provided axes. The resulted tensor has the same shape as the input if keepdims equal 1. 
  If keepdims equal 0, then the resulted tensor have the reduced dimension pruned. 
  The type of the output tensor is integer.
  * **attribute**:
    <dl>
      <dt>axes</dt>
      <dd>A list of integers, along which to reduce max.</dd>
      <dt>keepdims</dt>
      <dd>Keep the reduced dimension or not, default 1 mean keep reduced dimension.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>data</dt>
      <dd>An input tensor.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>reduced</dt>
      <dd>Reduced output tensor with integer data type.</dd>
    </dl>


* **ArgMin**

  Computes the indices of the min elements of the input tensor's element along the 
  provided axes. The resulted tensor has the same shape as the input if keepdims equal 1. 
  If keepdims equal 0, then the resulted tensor have the reduced dimension pruned. 
  The type of the output tensor is integer.
  * **attribute**:
    <dl>
      <dt>axes</dt>
      <dd>A list of integers, along which to reduce max.</dd>
      <dt>keepdims</dt>
      <dd>Keep the reduced dimension or not, default 1 mean keep reduced dimension.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>data</dt>
      <dd>An input tensor.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>reduced</dt>
      <dd>Reduced output tensor with integer data type.</dd>
    </dl>


* **AveragePool**

  AveragePool consumes an input tensor X and applies average pooling across the
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   Average pooling consisting of averaging all values of a subset of the
   input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing.
  * **attribute**:
    <dl>
      <dt>kernel_shape</dt>
      <dd>The size of the kernel along each axis.</dd>
      <dt>pads</dt>
      <dd>Padding along each axis, can take the value 0 (False) or non 0 (True)</dd>
      <dt>strides</dt>
      <dd>Stride along each axis.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimension are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.</dd>
    </dl>


* **BatchNormalization**

  Carries out batch normalization as described in the paper
  https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
  there are multiple cases for the number of outputs, which we list below:
  
  Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
  Output case #2: Y (test mode)
  * **attribute**:
    <dl>
      <dt>epsilon</dt>
      <dd>The epsilon value to use to avoid division by zero.</dd>
      <dt>is_test</dt>
      <dd>If set to nonzero, run spatial batch normalization in test mode.</dd>
      <dt>momentum</dt>
      <dd>Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum)</dd>
      <dt>spatial</dt>
      <dd>Compute the mean and variance across all spatial elements or per feature.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>The input 4-dimensional tensor of shape NCHW or NHWC depending on the order parameter.</dd>
      <dt>scale</dt>
      <dd>The scale as a 1-dimensional tensor of size C to be applied to the output.</dd>
      <dt>bias</dt>
      <dd>The bias as a 1-dimensional tensor of size C to be applied to the output.</dd>
      <dt>mean</dt>
      <dd>The running mean (training) or the estimated mean (testing) as a 1-dimensional tensor of size C.</dd>
      <dt>var</dt>
      <dd>The running variance (training) or the estimated variance (testing) as a 1-dimensional tensor of size C.</dd>
    </dl>
  * **output**:0 - &#8734;
    <dl>
      <dt>Y</dt>
      <dd>The output 4-dimensional tensor of the same shape as X.</dd>
      <dt>mean</dt>
      <dd>The running mean after the BatchNormalization operator. Must be in-place with the input mean. Should not be used for testing.</dd>
      <dt>var</dt>
      <dd>The running variance after the BatchNormalization operator. Must be in-place with the input var. Should not be used for testing.</dd>
      <dt>saved_mean</dt>
      <dd>Saved mean used during training to speed up gradient computation. Should not be used for testing.</dd>
      <dt>saved_var</dt>
      <dd>Saved variance used during training to speed up gradient computation. Should not be used for testing.</dd>
    </dl>


* **Cast**

  The operator casts the elements of a given input tensor to a data type
  specified by the 'to' argument and returns an output tensor of the same size in
  the converted type. The 'to' argument must be one of the data types specified
  in the 'DataType' enum field in the TensorProto message. If the 'to' argument
  is not provided or is not one of the enumerated types in DataType, Caffe2
  throws an Enforce error.
  
  NOTE: Casting to and from strings is not supported yet.
  * **attribute**:
    <dl>
      <dt>to</dt>
      <dd>The data type to which the elements of the input tensor are cast.Strictly must be one of the types from DataType enum in TensorProto</dd>
    </dl>
  * **input**:
    <dl>
      <dt>input</dt>
      <dd>Input tensor to be cast.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>Output tensor with the same shape as input with type specified by the 'to' argument</dd>
    </dl>


* **Ceil**

  Ceil takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the ceil is, y = ceil(x), is applied to
  the tensor elementwise.
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output tensor</dd>
    </dl>


* **Concat**

  Concatenate a list of tensors into a single tensor
  * **attribute**:
    <dl>
      <dt>axis</dt>
      <dd>Which axis to concat on</dd>
    </dl>
  * **input**:1 - &#8734;
  * **output**:
    <dl>
      <dt>concat_result</dt>
      <dd>Concatenated tensor</dd>
    </dl>


* **Constant**

  A constant tensor.
  * **attribute**:
    <dl>
      <dt>value</dt>
      <dd>The value for the elements of the output tensor.</dd>
    </dl>
  * **input**:
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>Output tensor containing the same value of the provided tensor.</dd>
    </dl>


* **Conv**

  The convolution operator consumes an input tensor and a filter, and
  computes the output.
  * **attribute**:
    <dl>
      <dt>dilations</dt>
      <dd>dilation value along each axis of the filter.</dd>
      <dt>group</dt>
      <dd>number of groups input channels and output channels are divided into</dd>
      <dt>kernel_shape</dt>
      <dd>The shape of the convolution kernel.</dd>
      <dt>pads</dt>
      <dd>Padding along each axis, can take the value 0 (False) or non 0 (True)</dd>
      <dt>strides</dt>
      <dd>stride along each axis.</dd>
    </dl>
  * **input**:2 - 3
    <dl>
      <dt>X</dt>
      <dd>Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image.Otherwise the size is (N x D1 x D2 ... x Dn)</dd>
      <dt>weights</dt>
      <dd>The weight tensor that will be used in the convolutions; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C x k1 x k2 x ... x kn), where is the dimension of the kernel</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.</dd>
    </dl>


* **ConvTranspose**

  The convolution transpose operator consumes an input tensor and a filter,
  and computes the output.
  * **attribute**:
    <dl>
      <dt>dilations</dt>
      <dd>dilation value along each axis of the filter.</dd>
      <dt>kernel_shape</dt>
      <dd>The shape of the convolution kernel.</dd>
      <dt>output_shape</dt>
      <dd>The shape of the output.</dd>
      <dt>pads</dt>
      <dd>Padding along each axis, can take the value 0 (False) or non 0 (True)</dd>
      <dt>strides</dt>
      <dd>stride along each axis.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image.Otherwise the size is (N x D1 x D2 ... x Dn)</dd>
      <dt>weights</dt>
      <dd>The weight tensor that will be used in the convolutions; has size (C x M x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (C x M x k1 x k2 x ... x kn), where is the dimension of the kernel</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.</dd>
    </dl>


* **Div**

  Performs element-wise binary division (with limited broadcast support).
  
  If necessary the right-hand-side argument will be broadcasted to match the
  shape of left-hand-side argument. When broadcasting is specified, the second
  tensor can either be of size 1 (a scalar value), or having its shape as a
  contiguous subset of the first tensor's shape. The starting of the mutually
  equal shape is specified by the argument "axis", and if it is not set, suffix
  matching is assumed. 1-dim expansion doesn't work yet.
  
  For example, the following tensor shapes are supported (with broadcast=1):
  
    shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
    shape(A) = (2, 3, 4, 5), shape(B) = (5,)
    shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
    shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
    shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
  
  Attribute `broadcast=1` needs to be passed to enable broadcasting.
  * **attribute**:
    <dl>
      <dt>axis</dt>
      <dd>If set, defines the broadcast dimensions. See doc for details.</dd>
      <dt>broadcast</dt>
      <dd>Pass 1 to enable broadcasting</dd>
    </dl>
  * **input**:
    <dl>
      <dt>A</dt>
      <dd>First operand, should share the type with the second operand.</dd>
      <dt>B</dt>
      <dd>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>C</dt>
      <dd>Result, has same dimensions and type as A</dd>
    </dl>


* **Dot**

  Apply dot product between 2 tensors. Similar to numpy implementation:
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor of any shape</dd>
      <dt>Y</dt>
      <dd>Input tensor of any shape</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Z</dt>
      <dd>Output tensor the dot product between X and Y.</dd>
    </dl>


* **Dropout**

  Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
  output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
  test mode or not, the output Y will either be a random dropout, or a simple
  copy of the input. Note that our implementation of Dropout does scaling in
  the training phase, so during testing nothing needs to be done.
  * **attribute**:
    <dl>
      <dt>is_test</dt>
      <dd>(int, default 0) if nonzero, run dropout in test mode where the output is simply Y = X.</dd>
      <dt>ratio</dt>
      <dd>(float, default 0.5) the ratio of random dropout</dd>
    </dl>
  * **input**:
    <dl>
      <dt>data</dt>
      <dd>The input data as Tensor.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>The output.</dd>
      <dt>mask</dt>
      <dd>The output mask. If is_test is nonzero, this output is not filled.</dd>
    </dl>


* **Elu**

  Elu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
  0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.
  
  * **attribute**:
    <dl>
      <dt>alpha</dt>
      <dd>Coefficient of ELU default to 1.0.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>1D input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>1D input tensor</dd>
    </dl>


* **Exp**

  Calculates the exponential of the given input tensor, element-wise. This
  operation can be done in an in-place fashion too, by providing the same input
  and output blobs.
  * **input**:
    <dl>
      <dt>input</dt>
      <dd>Input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>The exponential of the input tensor computed element-wise</dd>
    </dl>


* **Flatten**

  Flattens the input tensor into a 2D matrix, keeping the first dimension
  unchanged.
  * **input**:
    <dl>
      <dt>input</dt>
      <dd>A tensor of rank >= 2.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>A tensor of rank 2 with the contents of the input tensor, with first dimension equal first dimension of input, and remaining input dimensions flattened into the inner dimension of the output.</dd>
    </dl>


* **Floor**

  Floor takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the floor is, y = floor(x), is applied to
  the tensor elementwise.
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output tensor</dd>
    </dl>


* **Gather**

  Given DATA tensor of rank r >= 1, and INDICES tensor of rank q, gather
  entries of the outer-most dimension of DATA indexed by INDICES, and concatenate
  them in an output tensor of rank q + (r - 1).
  
  Example:
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [
        [0, 1],
        [1, 2],
    ]
    OUTPUT = [
        [
            [1.0, 1.2],
            [2.3, 3.4],
        ],
        [
            [2.3, 3.4],
            [4.5, 5.7],
        ],
    ]
  * **input**:
    <dl>
      <dt>DATA</dt>
      <dd>Tensor of rank r >= 1.</dd>
      <dt>INDICES</dt>
      <dd>Tensor of int32/int64 indices, of any rank q.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>OUTPUT</dt>
      <dd>Tensor of rank q + (r - 1).</dd>
    </dl>


* **GlobalAveragePool**

  GlobalAveragePool consumes an input tensor X and applies average pooling across the
   the values in the same channel. This is equivalent to AveragePool with kernel size
   equal to the spatial dimension of input tensor.
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimension are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output data tensor from pooling across the input tensor. Dimensions will be N x C x 1 x 1</dd>
    </dl>


* **GlobalMaxPool**

  GlobalMaxPool consumes an input tensor X and applies max pooling across the
   the values in the same channel. This is equivalent to MaxPool with kernel size
   equal to the spatial dimension of input tensor.
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimension are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output data tensor from pooling across the input tensor. Dimensions will be N x C x 1 x 1</dd>
    </dl>


* **LeakyRelu**

  LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
  output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`, is applied to the data tensor elementwise.
  * **attribute**:
    <dl>
      <dt>alpha</dt>
      <dd>Coefficient of leakage</dd>
    </dl>
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output tensor</dd>
    </dl>


* **Log**

  Calculates the natural log of the given input tensor, element-wise. This
  operation can be done in an in-place fashion too, by providing the same input
  and output blobs.
  * **input**:
    <dl>
      <dt>input</dt>
      <dd>Input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>The natural log of the input tensor computed element-wise</dd>
    </dl>


* **Max**

  Element-wise max of each of the input tensors. The first input tensor can be
  used in-place as the output tensor, in which case the max will be done in
  place and results will be accumulated in input0. All inputs and outputs must
  have the same shape and data type.
  * **input**:1 - &#8734;
    <dl>
      <dt>data_0</dt>
      <dd>First of the input tensors. Can be inplace.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>max</dt>
      <dd>Output tensor. Same dimension as inputs.</dd>
    </dl>


* **MaxPool**

  MaxPool consumes an input tensor X and applies max pooling across the
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   Average pooling consisting of averaging all values of a subset of the
   input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing.
  * **attribute**:
    <dl>
      <dt>dilations</dt>
      <dd>Dilation along each axis, 1 mean no dilation.</dd>
      <dt>kernel_shape</dt>
      <dd>The size of the kernel along each axis.</dd>
      <dt>pads</dt>
      <dd>Padding along each axis, can take the value 0 (False) or non 0 (True)</dd>
      <dt>strides</dt>
      <dd>Stride along each axis.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimension are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output data tensor from max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.</dd>
    </dl>


* **Min**

  Element-wise min of each of the input tensors. The first input tensor can be
  used in-place as the output tensor, in which case the max will be done in
  place and results will be accumulated in input0. All inputs and outputs must
  have the same shape and data type.
  * **input**:1 - &#8734;
    <dl>
      <dt>data_0</dt>
      <dd>First of the input tensors. Can be inplace.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>max</dt>
      <dd>Output tensor. Same dimension as inputs.</dd>
    </dl>


* **Mul**

  Performs element-wise binary multiplication (with limited broadcast support).
  
  If necessary the right-hand-side argument will be broadcasted to match the
  shape of left-hand-side argument. When broadcasting is specified, the second
  tensor can either be of size 1 (a scalar value), or having its shape as a
  contiguous subset of the first tensor's shape. The starting of the mutually
  equal shape is specified by the argument "axis", and if it is not set, suffix
  matching is assumed. 1-dim expansion doesn't work yet.
  
  For example, the following tensor shapes are supported (with broadcast=1):
  
    shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
    shape(A) = (2, 3, 4, 5), shape(B) = (5,)
    shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
    shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
    shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
  
  Attribute `broadcast=1` needs to be passed to enable broadcasting.
  * **attribute**:
    <dl>
      <dt>axis</dt>
      <dd>If set, defines the broadcast dimensions. See doc for details.</dd>
      <dt>broadcast</dt>
      <dd>Pass 1 to enable broadcasting</dd>
    </dl>
  * **input**:
    <dl>
      <dt>A</dt>
      <dd>First operand, should share the type with the second operand.</dd>
      <dt>B</dt>
      <dd>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>C</dt>
      <dd>Result, has same dimensions and type as A</dd>
    </dl>


* **Neg**

  Neg takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where each element flipped sign, y = -x, is applied to
  the tensor elementwise.
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output tensor</dd>
    </dl>


* **PRelu**

  PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
  output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
  `f(x) = x for x >= 0`., is applied to the data tensor elementwise.
  
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor</dd>
      <dt>Slope</dt>
      <dd>Slope tensor. If `Slope` is of size 1, the value is shared across different channels</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Input tensor</dd>
    </dl>


* **Pow**

  Pow takes input data (Tensor<T>) and an argument exponent, and
  produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
  is applied to the data tensor elementwise.
  * **attribute**:
    <dl>
      <dt>exponent</dt>
      <dd>The exponent of the power function.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor of any shape</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output tensor (same size as X)</dd>
    </dl>


* **RandomNormal**

  Generate a tensor with random values drawn from a normal distribution. The shape
  of the tensor is specified by the `shape` argument and the parameter of the normal distribution
  specified by `mean` and `scale`.
  
  The data type is specified by the 'dtype' argument. The 'dtype' argument must
  be one of the data types specified in the 'DataType' enum field in the
  TensorProto message.
  * **attribute**:
    <dl>
      <dt>dtype</dt>
      <dd>The data type for the elements of the output tensor.</dd>
      <dt>mean</dt>
      <dd>The mean of the normal distribution.</dd>
      <dt>scale</dt>
      <dd>The standard deviation of the normal distribution.</dd>
      <dt>seed</dt>
      <dd>(Optional) Seed to the random generator, if not specified we will auto generate one.</dd>
      <dt>shape</dt>
      <dd>The shape of the output tensor.</dd>
    </dl>
  * **input**:
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>Output tensor of random values drawn from normal distribution</dd>
    </dl>


* **RandomNormalLike**

  Generate a tensor with random values drawn from a normal distribution. The shape
  of the tensor is computed from the input argument and the parameter of the normal distribution
  specified by `mean` and `scale`.
  
  The data type is specified by the 'dtype' argument. The 'dtype' argument must
  be one of the data types specified in the 'DataType' enum field in the
  TensorProto message.
  * **attribute**:
    <dl>
      <dt>dtype</dt>
      <dd>(Optional) The data type for the elements of the output tensor, if not specified, we will usethe data type of the input tensor.</dd>
      <dt>mean</dt>
      <dd>The mean of the normal distribution.</dd>
      <dt>scale</dt>
      <dd>The standard deviation of the normal distribution.</dd>
      <dt>seed</dt>
      <dd>(Optional) Seed to the random generator, if not specified we will auto generate one.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>input</dt>
      <dd>Input tensor to provide shape information.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>Output tensor of random values drawn from normal distribution</dd>
    </dl>


* **RandomUniform**

  Generate a tensor with random values drawn from a uniform distribution. The shape
  of the tensor is specified by the `shape` argument and the range by `low` and `high`.
  
  The data type is specified by the 'dtype' argument. The 'dtype' argument must
  be one of the data types specified in the 'DataType' enum field in the
  TensorProto message.
  * **attribute**:
    <dl>
      <dt>dtype</dt>
      <dd>The data type for the elements of the output tensor.</dd>
      <dt>high</dt>
      <dd>Upper boundary of the output values.</dd>
      <dt>low</dt>
      <dd>Lower boundary of the output values.</dd>
      <dt>seed</dt>
      <dd>(Optional) Seed to the random generator, if not specified we will auto generate one.</dd>
      <dt>shape</dt>
      <dd>The shape of the output tensor.</dd>
    </dl>
  * **input**:
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>Output tensor of random values drawn from uniform distribution</dd>
    </dl>


* **RandomUniformLike**

  Generate a tensor with random values drawn from a uniform distribution. The shape
  of the tensor is computed from the input argument and the range by `low` and `high`.
  
  The data type is specified by the 'dtype' argument. The 'dtype' argument must
  be one of the data types specified in the 'DataType' enum field in the
  TensorProto message.
  * **attribute**:
    <dl>
      <dt>dtype</dt>
      <dd>(Optional) The data type for the elements of the output tensor, if not specified, we will usethe data type of the input tensor.</dd>
      <dt>high</dt>
      <dd>Upper boundary of the output values.</dd>
      <dt>low</dt>
      <dd>Lower boundary of the output values.</dd>
      <dt>seed</dt>
      <dd>(Optional) Seed to the random generator, if not specified we will auto generate one.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>input</dt>
      <dd>Input tensor to provide shape information.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>Output tensor of random values drawn from uniform distribution</dd>
    </dl>


* **Reciprocal**

  Reciprocal takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the reciprocal is, y = 1/x, is applied to
  the tensor elementwise.
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output tensor</dd>
    </dl>


* **ReduceLogSumExp**

  Computes the log sum exponent of the input tensor's element along the provided axes. The resulted
  tensor has the same shape as the input if keepdims equal 1. If keepdims equal 0, then 
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  * **attribute**:
    <dl>
      <dt>axes</dt>
      <dd>A list of integers, along which to reduce max.</dd>
      <dt>keepdims</dt>
      <dd>Keep the reduced dimension or not, default 1 mean keep reduced dimension.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>data</dt>
      <dd>An input tensor.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>reduced</dt>
      <dd>Reduced output tensor.</dd>
    </dl>


* **ReduceMax**

  Computes the max of the input tensor's element along the provided axes. The resulted
  tensor has the same shape as the input if keepdims equal 1. If keepdims equal 0, then 
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  * **attribute**:
    <dl>
      <dt>axes</dt>
      <dd>A list of integers, along which to reduce max.</dd>
      <dt>keepdims</dt>
      <dd>Keep the reduced dimension or not, default 1 mean keep reduced dimension.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>data</dt>
      <dd>An input tensor.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>reduced</dt>
      <dd>Reduced output tensor.</dd>
    </dl>


* **ReduceMean**

  Computes the mean of the input tensor's element along the provided axes. The resulted
  tensor has the same shape as the input if keepdims equal 1. If keepdims equal 0, then 
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  * **attribute**:
    <dl>
      <dt>axes</dt>
      <dd>A list of integers, along which to reduce max.</dd>
      <dt>keepdims</dt>
      <dd>Keep the reduced dimension or not, default 1 mean keep reduced dimension.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>data</dt>
      <dd>An input tensor.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>reduced</dt>
      <dd>Reduced output tensor.</dd>
    </dl>


* **ReduceMin**

  Computes the min of the input tensor's element along the provided axes. The resulted
  tensor has the same shape as the input if keepdims equal 1. If keepdims equal 0, then 
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  * **attribute**:
    <dl>
      <dt>axes</dt>
      <dd>A list of integers, along which to reduce max.</dd>
      <dt>keepdims</dt>
      <dd>Keep the reduced dimension or not, default 1 mean keep reduced dimension.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>data</dt>
      <dd>An input tensor.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>reduced</dt>
      <dd>Reduced output tensor.</dd>
    </dl>


* **ReduceProd**

  Computes the product of the input tensor's element along the provided axes. The resulted
  tensor has the same shape as the input if keepdims equal 1. If keepdims equal 0, then 
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  * **attribute**:
    <dl>
      <dt>axes</dt>
      <dd>A list of integers, along which to reduce max.</dd>
      <dt>keepdims</dt>
      <dd>Keep the reduced dimension or not, default 1 mean keep reduced dimension.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>data</dt>
      <dd>An input tensor.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>reduced</dt>
      <dd>Reduced output tensor.</dd>
    </dl>


* **ReduceSum**

  Computes the sum of the input tensor's element along the provided axes. The resulted
  tensor has the same shape as the input if keepdims equal 1. If keepdims equal 0, then 
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  * **attribute**:
    <dl>
      <dt>axes</dt>
      <dd>A list of integers, along which to reduce max.</dd>
      <dt>keepdims</dt>
      <dd>Keep the reduced dimension or not, default 1 mean keep reduced dimension.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>data</dt>
      <dd>An input tensor.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>reduced</dt>
      <dd>Reduced output tensor.</dd>
    </dl>


* **Relu**

  Relu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
  the tensor elementwise.
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output tensor</dd>
    </dl>


* **Reshape**

  Reshape the input tensor similar to numpy.reshape.
      
  It takes a tensor as input and an argument `shape`. It outputs the reshaped tensor.
      
  At most one dimension of the new shape can be -1. In this case, the value is
  inferred from the size of the tensor and the remaining dimensions. A dimension
  could also be 0, in which case the actual dimension value is going to be copied
  from the shape argument.
  * **attribute**:
    <dl>
      <dt>shape</dt>
      <dd>New shape</dd>
    </dl>
  * **input**:
    <dl>
      <dt>data</dt>
      <dd>An input tensor.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>reshaped</dt>
      <dd>Reshaped data.</dd>
    </dl>


* **Selu**

  Selu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the scaled exponential linear unit function,
  `y = gamma * (alpha * e^x - alpha) for x <= 0`, `f(x) = gamma * x for x > 0`,
  is applied to the tensor elementwise.
  * **attribute**:
    <dl>
      <dt>alpha</dt>
      <dd>Coefficient of SELU default to 1.6732.</dd>
      <dt>gamma</dt>
      <dd>Coefficient of SELU default to 1.0507.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output tensor</dd>
    </dl>


* **Sigmoid**

  Sigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
  tensor elementwise.
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output tensor</dd>
    </dl>


* **Slice**

  Produces a slice of the input tensor along multiple axes. Similar to numpy:
  https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html 
  
  Slices are passed as two keyword argument lists with starting and end indices 
  for each dimension of the input `data` tensor. If a negative value is passed 
  for any of the start or end indices, it represent number of elements before 
  the end of that dimension.
  
  `strides` is the  step sizes when applying slicing, negative value means in 
  reverse order.
  * **attribute**:
    <dl>
      <dt>ends</dt>
      <dd>List of ending indices</dd>
      <dt>starts</dt>
      <dd>List of starting indices</dd>
    </dl>
  * **input**:1 - 3
    <dl>
      <dt>data</dt>
      <dd>Tensor of data to extract slices from.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>Sliced data tensor.</dd>
    </dl>


* **Softmax**

  The operator computes the softmax normalized values for each layer in the batch
   of the given input. The input is a 2-D tensor (Tensor<float>) of size
  (batch_size x input_feature_dimensions). The output tensor has the same shape
  and contains the softmax normalized values of the corresponding input.
  
  X does not need to explicitly be a 2D vector; rather, it will be
  coerced into one. For an arbitrary n-dimensional tensor
  X \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
  the axis provided, then X will be coerced into a 2-dimensional tensor with
  dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
  case where axis=1, this means the X tensor will be coerced into a 2D tensor
  of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
  In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
  Each of these dimensions must be matched correctly, or else the operator
  will throw errors.
  * **attribute**:
    <dl>
      <dt>axis</dt>
      <dd>(int) default to 1; describes the axis of the inputs when coerced to 2D; defaults to one because the 0th axis most likely describes the batch_size</dd>
    </dl>
  * **input**:
    <dl>
      <dt>input</dt>
      <dd>The input tensor that's coerced into a 2D matrix of size (NxD) as described above.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>The softmax normalized output values with the same shape as input tensor.</dd>
    </dl>


* **Split**

  Split a tensor into a list of tensors, along the specified
  'axis'. The lengths of the split can be specified using argument 'axis' or
  optional second input blob to the operator. Otherwise, the tensor is split
  to equal sized parts.
  * **attribute**:
    <dl>
      <dt>axis</dt>
      <dd>Which axis to split on</dd>
      <dt>split</dt>
      <dd>length of each output</dd>
    </dl>
  * **input**:1 - 2
    <dl>
      <dt>input</dt>
      <dd>The tensor to split</dd>
      <dt>split</dt>
      <dd>Optional list of output lengths (see also arg 'split')</dd>
    </dl>
  * **output**:1 - &#8734;


* **Sqrt**

  Square root takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the square root is, y = x^0.5, is applied to
  the tensor elementwise. If x is negative, then it will return NaN.
  * **input**:
    <dl>
      <dt>X</dt>
      <dd>Input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>Y</dt>
      <dd>Output tensor</dd>
    </dl>


* **Squeeze**

  Remove single-dimensional entries from the shape of a tensor.
  Takes a  parameter `axes` with a list of axes to squeeze.
  * **attribute**:
    <dl>
      <dt>axes</dt>
      <dd>List of positive integers, indicate the dimensions to squeeze.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>data</dt>
      <dd>Tensors with at least max(dims) dimensions.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>squeezed</dt>
      <dd>Reshaped tensor with same data as input.</dd>
    </dl>


* **Sub**

  Performs element-wise binary subtraction (with limited broadcast support).
  
  If necessary the right-hand-side argument will be broadcasted to match the
  shape of left-hand-side argument. When broadcasting is specified, the second
  tensor can either be of size 1 (a scalar value), or having its shape as a
  contiguous subset of the first tensor's shape. The starting of the mutually
  equal shape is specified by the argument "axis", and if it is not set, suffix
  matching is assumed. 1-dim expansion doesn't work yet.
  
  For example, the following tensor shapes are supported (with broadcast=1):
  
    shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
    shape(A) = (2, 3, 4, 5), shape(B) = (5,)
    shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
    shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
    shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
  
  Attribute `broadcast=1` needs to be passed to enable broadcasting.
  * **attribute**:
    <dl>
      <dt>axis</dt>
      <dd>If set, defines the broadcast dimensions. See doc for details.</dd>
      <dt>broadcast</dt>
      <dd>Pass 1 to enable broadcasting</dd>
    </dl>
  * **input**:
    <dl>
      <dt>A</dt>
      <dd>First operand, should share the type with the second operand.</dd>
      <dt>B</dt>
      <dd>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>C</dt>
      <dd>Result, has same dimensions and type as A</dd>
    </dl>


* **Sum**

  Element-wise sum of each of the input tensors. The first input tensor can be
  used in-place as the output tensor, in which case the sum will be done in
  place and results will be accumulated in input0. All inputs and outputs must
  have the same shape and data type.
  * **input**:1 - &#8734;
    <dl>
      <dt>data_0</dt>
      <dd>First of the input tensors. Can be inplace.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>sum</dt>
      <dd>Output tensor. Same dimension as inputs.</dd>
    </dl>


* **Tanh**

  Calculates the hyperbolic tangent of the given input tensor element-wise. This
  operation can be done in an in-place fashion too, by providing the same input
  and output blobs.
  * **input**:
    <dl>
      <dt>input</dt>
      <dd>1-D input tensor</dd>
    </dl>
  * **output**:
    <dl>
      <dt>output</dt>
      <dd>The hyperbolic tangent values of the input tensor computed element-wise</dd>
    </dl>


* **Transpose**

  Transpose the input tensor similar to numpy.transpose. For example, when
  axes=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
  will be (2, 1, 3).
  * **attribute**:
    <dl>
      <dt>perm</dt>
      <dd>A list of integers. By default, reverse the dimensions, otherwise permute the axes according to the values given.</dd>
    </dl>
  * **input**:
    <dl>
      <dt>data</dt>
      <dd>An input tensor.</dd>
    </dl>
  * **output**:
    <dl>
      <dt>transposed</dt>
      <dd>Transposed output.</dd>
    </dl>


