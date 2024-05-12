import haiku as hk
import jax.numpy as jnp
import jax

from typing import Any, Callable, Optional, Sequence, Union
from jaxtyping import Array

from haiku._src.conv import compute_adjusted_padding


class UNetND(hk.Module):
    def __init__(
        self,
        num_spatial_dim: int,
        out_channels: Sequence[int] = [32, 64, 128],
        kernel_shape: Union[int, Sequence[int]] = 4,
        stride: Union[int, Sequence[int]] = 2,
        num_groups: int = 4,
        kernel_shape_resnet: Union[int, Sequence[int]] = 3,
        stride_resnet: Union[int, Sequence[int]] = 1,
        with_bias: bool = True,
        with_attention: bool = True,
        act: Callable = jax.nn.silu,
    ):
        """This is a general UNet implementation with ResNet blocks and optional attention layers.

        The input to this function should be of shape [*batch_sizes, *spatial_dims, channels_in], where len(spatial_dims) == num_spatial_dims
        The optional context should be of shape [batch_size, context_dim]
        The output of this function will be of shape [*batch_sizes, *spatial_dims, channels_out], where len(spatial_dims) == num_spatial_dims

        This will be a symmetric UNet, with the same number of downsampling and upsampling layers.
        At each downsampling layer, the number of channels will increased as specified by out_channels.
        At each downsampling layer, the number of dimensions will be reduced by the given stride.
        At each upsampling layer, the number of channels will decreased as specified by out_channels.
        At each upsampling layer, the number of dimensions will be increased by the given stride.


        Args:
            num_spatial_dim (int): Spatial dimensionality of the input.
            out_channels (Sequence[int], optional): Number of output channels generated during downsampling and reversed during upsampling. Defaults to [32, 64, 128].
            kernel_shape (Union[int, Sequence[int]], optional): Kernel shape of down and upsampling layers. Defaults to 4.
            stride (Union[int, Sequence[int]], optional): Stride. Defaults to 2.
            num_groups (int, optional): Numer of groups, for group normalization. If used with with_attention, each group will get its own attention head! . Defaults to 4.
            kernel_shape_resnet (Union[int, Sequence[int]], optional): Kernel shape of ResNet blocks. Defaults to 3.
            stride_resnet (Union[int, Sequence[int]], optional): Stride of ResNet blocks. Defaults to 1.
            with_bias (bool, optional): If a bias should be used in down and upsampling convolution. Defaults to True.
            with_attention (bool, optional): If attention should be used after ResNet blocks. Defaults to True.
        """
        super().__init__()
        self.num_spaital_dim = num_spatial_dim
        self.out_channels = out_channels
        self.num_stages = len(out_channels)
        self.kernel_shape = kernel_shape
        self.kernel_shape_resnet = kernel_shape_resnet
        self.stride = stride
        self.stride_resnet = stride_resnet
        self.num_groups = num_groups
        self.act = act
        self.with_bias = with_bias
        self.with_attention = with_attention

        assert len(out_channels) >= 2, "Must have at least 2 output channels"
        assert all(
            o % self.num_groups == 0 for o in out_channels
        ), "Output channels must be divisible by num_groups!"

    def __call__(self, inputs: Array, context: Optional[Array] = None):
        channels = inputs.shape[-1]
        spatial_dims = inputs.shape[-self.num_spaital_dim - 1 : -1]
        batch_dims = inputs.shape[: -self.num_spaital_dim - 1]

        kernel_shape = self.kernel_shape
        stride_shape = self.stride

        # Initial convolutional layer, with larger kernel size
        x = hk.ConvND(
            self.num_spaital_dim,
            self.out_channels[0],
            kernel_shape=kernel_shape + 3
            if isinstance(kernel_shape, int)
            else [k + 3 for k in kernel_shape],
            padding="SAME",
            with_bias=self.with_bias,
        )(inputs)
        if context is not None:
            context_dim = context.shape[-1]
            context_emb = jnp.broadcast_to(context, batch_dims + (context_dim,))

        pre_downsampling = []

        # Downsampling phase
        for index, num_out_channel in enumerate(self.out_channels):
            print(x.shape)
            # ResNet blocks
            x = self.resnet_block(x, num_out_channel, context_emb)
            x = self.resnet_block(x, num_out_channel, context_emb)

            # Attention layer
            if self.with_attention:
                att = self.attention(x)
                x = x + att

            # Downsample with strided convolution
            # Saving this output for residual connection with the upsampling layer
            pre_downsampling.append(x)
            if index != self.num_stages - 1:
                x = hk.ConvND(
                    self.num_spaital_dim,
                    self.out_channels[index + 1],
                    kernel_shape=kernel_shape,
                    stride=stride_shape,
                    padding="SAME",
                    with_bias=self.with_bias,
                )(x)

        # Middle block
        x = self.resnet_block(x, self.out_channels[-1], context_emb)
        if self.with_attention:
            att = self.attention(x)
            x = x + att
        x = self.resnet_block(x, self.out_channels[-1], context_emb)

        # Upsampling phase
        for index, num_out_channel in enumerate(reversed(self.out_channels)):
            # Concatenate with output from downsampling phase
            x = jnp.concatenate([pre_downsampling.pop(), x], -1)

            # ResNet blocks
            x = self.resnet_block(x, num_out_channel, context_emb)
            x = self.resnet_block(x, num_out_channel, context_emb)

            if self.with_attention:
                att = self.attention(x)
                x = x + att

            # Upsample with transposed convolution
            if index != len(self.out_channels) - 1:
                # Gurantee that the output is the downsampled input
                padding = self._get_padding(
                    input_size=x.shape[1],
                    output_size=pre_downsampling[-1].shape[1],
                )

                x = hk.ConvNDTranspose(
                    self.num_spaital_dim,
                    self.out_channels[-index - 2],
                    kernel_shape=kernel_shape,
                    stride=stride_shape,
                    padding=[padding for _ in spatial_dims],
                    with_bias=self.with_bias,
                )(x)

        # Final ResNet block and output convolutional layer
        x = self.resnet_block(x, self.out_channels[0], context_emb)
        x = hk.ConvND(
            self.num_spaital_dim,
            channels,
            kernel_shape=1,
            padding="SAME",
            with_bias=self.with_bias,
        )(x)
        return x

    @hk.transparent
    def _get_padding(self, input_size, output_size):
        padding = compute_adjusted_padding(
            input_size=input_size,
            output_size=output_size,
            kernel_size=self.kernel_shape,
            stride=self.stride,
            padding="SAME",
        )
        return padding

    @hk.transparent
    def resnet_block(
        self, inputs: Array, num_out_channel: int, context_emb: Optional[Array]
    ):
        block = ResnetBlock(
            self.num_spaital_dim,
            num_out_channel,
            num_groups=self.num_groups,
            kernel_shape=self.kernel_shape_resnet,
            stride=self.stride_resnet,
            act=self.act,
        )

        return block(inputs, context_emb)

    @hk.transparent
    def attention(self, inputs: Array):
        initializer = hk.initializers.VarianceScaling(1 / len(self.out_channels))
        attention_layer = hk.MultiHeadAttention(
            num_heads=self.num_groups,
            key_size=inputs.shape[-1] // self.num_groups,
            model_size=inputs.shape[-1],
            w_init=initializer,
        )
        norm_layer = hk.GroupNorm(self.num_groups)

        x = jnp.reshape(inputs, (inputs.shape[0], -1, inputs.shape[-1]))
        att = attention_layer(x, x, x)
        att = att.reshape(inputs.shape)
        out = norm_layer(att)

        return out


class UNet1D(UNetND):
    def __init__(
        self,
        out_channels: Sequence[int] = [32, 64, 128],
        kernel_shape: int | Sequence[int] = 4,
        stride: int | Sequence[int] = 2,
        num_groups: int = 4,
        kernel_shape_resnet: int | Sequence[int] = 3,
        stride_resnet: int | Sequence[int] = 1,
        with_bias: bool = True,
        with_attention: bool = True,
        act: Callable[..., Any] = jax.nn.silu,
    ):
        super().__init__(
            1,
            out_channels,
            kernel_shape,
            stride,
            num_groups,
            kernel_shape_resnet,
            stride_resnet,
            with_bias,
            with_attention,
            act,
        )


class UNet2D(UNetND):
    def __init__(
        self,
        out_channels: Sequence[int] = [32, 64, 128],
        kernel_shape: int | Sequence[int] = 4,
        stride: int | Sequence[int] = 2,
        num_groups: int = 4,
        kernel_shape_resnet: int | Sequence[int] = 3,
        stride_resnet: int | Sequence[int] = 1,
        with_bias: bool = True,
        with_attention: bool = True,
        act: Callable[..., Any] = jax.nn.silu,
    ):
        super().__init__(
            2,
            out_channels,
            kernel_shape,
            stride,
            num_groups,
            kernel_shape_resnet,
            stride_resnet,
            with_bias,
            with_attention,
            act,
        )


class ConvNDBlock(hk.Module):
    def __init__(
        self,
        num_spatial_dims: int,
        output_channels: int,
        kernel_shape: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        num_groups: Optional[int] = 8,
        act: Callable = jax.nn.silu,
    ):
        """This simple block consists of a convolutional layer, a normalization layer and an activation function.

        The input to this function should be of shape [*batch_sizes, *spatial_dims, channels_in], where len(spatial_dims) == num_spatial_dims
        The output of this function will be of shape [*batch_sizes, *spatial_dims, channels_out], where len(spatial_dims) == num_spatial_dims

        Args:
            num_spatial_dims (int): Spatial dimensionality of the input.
            output_channels (int): Output channels of the convolutional layer.
            kernel_shape (Sequence[int], optional): Shape of the kernel . Defaults to (3,).
            stride (Union[int, Sequence[int]], optional): Stride of the convolutional layer. Defaults to 1.
            num_groups (Optional[int], optional): Number of groups for GroupNormalization, if None then no normalization is performed. Defaults to 8.
            act (Callable, optional): Activation function. Defaults to jax.nn.silu.
            act (Callable, optional): Activation function. Defaults to jax.nn.silu.
        """
        super().__init__()
        self.num_spatial_dims = num_spatial_dims
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.num_groups = num_groups
        self.act = act

    def __call__(self, inputs):
        conv = hk.ConvND(
            self.num_spatial_dims,
            self.output_channels,
            kernel_shape=self.kernel_shape,
            stride=self.stride,
            padding="SAME",
        )(inputs)

        if self.num_groups is not None:
            norm = hk.GroupNorm(self.num_groups)(conv)
        activation = self.act(norm)
        return activation


class ResnetBlock(hk.Module):
    def __init__(
        self,
        num_spatial_dims: int,
        output_channels: int,
        kernel_shape: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        num_groups: Optional[int] = 8,
        act=jax.nn.silu,
    ):
        """This block consists of two convolutional layers with a residual connection. If a context is provided, it is added to the output of the first convolutional layer.

        The input to this function should be of shape [*batch_sizes, *spatial_dims, channels_in], where len(spatial_dims) == num_spatial_dims
        The optional context should be of shape [batch_size, context_dim]
        The output of this function will be of shape [*batch_sizes, *spatial_dims, channels_out], where len(spatial_dims) == num_spatial_dims

        Args:
            num_spatial_dims (int): _description_
            output_channels (int): _description_
            kernel_shape (Union[int, Sequence[int]], optional): _description_. Defaults to 3.
            stride (Union[int, Sequence[int]], optional): _description_. Defaults to 1.
            num_groups (Optional[int], optional): _description_. Defaults to 8.
            act (_type_, optional): _description_. Defaults to jax.nn.silu.
        """
        super().__init__()
        self.num_spatial_dims = num_spatial_dims
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.num_groups = num_groups
        self.act = act

    def __call__(self, inputs: Array, context: Optional[Array] = None):
        # First convolutional layer
        x = ConvNDBlock(
            self.num_spatial_dims,
            self.output_channels,
            self.kernel_shape,
            self.stride,
            self.num_groups,
            self.act,
        )(inputs)

        # Add context if provided
        if context is not None:
            context = hk.Linear(self.output_channels)(context)
            context = self.act(context)
            while context.ndim < x.ndim:
                context = context[..., None, :]
            x = x + context

        # Second convolutional layer
        x = ConvNDBlock(
            self.num_spatial_dims,
            self.output_channels,
            self.kernel_shape,
            self.stride,
            self.num_groups,
            self.act,
        )(x)

        # Residual connection
        skip_connection = hk.ConvND(
            self.num_spatial_dims,
            self.output_channels,
            (1, 1),
            padding="SAME",
            with_bias=False,
        )(
            inputs
        )  # This is required to match output_channels
        out = x + skip_connection
        return out
