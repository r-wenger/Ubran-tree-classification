# models.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from encoder import Encoder
from decoder import Decoder
from utils import generate_original_PE
import params

from typing import Optional, Any
import math
from torch import Tensor
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer, TransformerEncoder


"""
____________________________________________________________________________________
Credits:

The InceptionTime model has been inspired by the following paper:
'InceptionTime: Finding AlexNet for Time Series Classification': https://arxiv.org/abs/1909.04939

The implementation translation of the InceptionTime model in PyTorch has been inspired by the following repository:
https://github.com/TheMrGhostman/InceptionTime-Pytorch/tree/master

The Hybrid IncetionTime model has been inspired by the following paper:
'Deep Learning For Time Series Classification Using New Hand-Crafted Convolution Filters': https://doi.org/10.1109/BigData55660.2022.10020496
and the following repository originally implemented in Keras:
https://github.com/MSD-IRIMAS/CF-4-TSC/tree/main
____________________________________________________________________________________
"""

def pass_through(X):
	return X

class Flatten(nn.Module):
	def __init__(self, out_features):
		super(Flatten, self).__init__()
		self.output_dim = out_features

	def forward(self, x):
		return x.view(-1, self.output_dim)
    
class Reshape(nn.Module):
	def __init__(self, out_shape):
		super(Reshape, self).__init__()
		self.out_shape = out_shape

	def forward(self, x):
		return x.view(-1, *self.out_shape)

class Inception(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[10, 20, 40], bottleneck_channels=32, activation=nn.ReLU(), use_hybrid=False, return_indices=False):
        """
		: param in_channels				Number of input channels (input features)
		: param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
		: param kernel_sizes			List of kernel sizes for each convolution.
										Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
										This is necessary because of padding size.
										For correction of kernel_sizes use function "correct_sizes". 
		: param bottleneck_channels		Number of output channels in bottleneck. 
										Bottleneck wont be used if number of in_channels is equal to 1.
		: param activation				Activation function for output tensor (nn.ReLU()). 
		: param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d. 
        """
        super(Inception, self).__init__()
        self.return_indices=return_indices
        self.use_hybrid = use_hybrid
        
        # bottleneck layer of bottleneck_channels=32 filters of size 1
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                                in_channels=in_channels, 
                                out_channels=bottleneck_channels, 
                                kernel_size=1, 
                                stride=1, 
                                bias=False
                                )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        # the 3 convolutional layers with n_filters=32 of size kernel_sizes= 40, 20, 10
        self.conv_from_bottleneck_1 = nn.Conv1d(
                                        in_channels=bottleneck_channels, 
                                        out_channels=n_filters, 
                                        kernel_size=kernel_sizes[0], 
                                        stride=1, 
                                        padding='same', 
                                        bias=False
                                        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
                                        in_channels=bottleneck_channels, 
                                        out_channels=n_filters, 
                                        kernel_size=kernel_sizes[1], 
                                        stride=1, 
                                        padding='same', 
                                        bias=False
                                        )
        self.conv_from_bottleneck_3 = nn.Conv1d(
                                        in_channels=bottleneck_channels, 
                                        out_channels=n_filters, 
                                        kernel_size=kernel_sizes[2], 
                                        stride=1, 
                                        padding='same', 
                                        bias=False
                                        )

        # maxpooling layer with kernel size k=3
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)

        # convolutional layer after maxpooling layer of size bottleneck_channels=32 and kernel_size=1
        self.conv_from_maxpool = nn.Conv1d(
                                    in_channels=in_channels, 
                                    out_channels=n_filters, 
                                    kernel_size=1, 
                                    stride=1,
                                    padding=0, 
                                    bias=False
                                    )

        # hybrid layer (if use_hybrid=True) and batch normalization layer
        if use_hybrid:
            self.hybrid = HybridLayer(input_channels=bottleneck_channels)
            self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters+17)
        else:
            self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)

        # activation function
        self.activation = activation

    def forward(self, X):
        # step 1
        Z_bottleneck = self.bottleneck(X)
        if self.return_indices:
            Z_maxpool, indices = self.max_pool(X)
        else:
            Z_maxpool = self.max_pool(X)

        # step 2
        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        Z4 = self.conv_from_maxpool(Z_maxpool)
        
        # step 3
        if self.use_hybrid:
            Z5 = self.hybrid(Z_bottleneck)
            Z = torch.cat([Z1, Z2, Z3, Z4, Z5], dim=1) # 4*n_filters+17 filters as output
        else:
            Z = torch.cat([Z1, Z2, Z3, Z4], dim=1) # 4*n_filters as output
        Z = self.activation(self.batch_norm(Z))
        # return indices if needed for decoder
        if self.return_indices:
            return Z, indices
        else:
            return Z


class InceptionBlock(nn.Module):   
	def __init__(self, in_channels, n_filters=32, kernel_sizes=[10,20,40], bottleneck_channels=32, use_residual=True, activation=nn.ReLU(), use_hybrid=False, return_indices=False):
		"""
        # It is creating 1 of the 2 blocks of the InceptionTime architecture
		"""
		super(InceptionBlock, self).__init__()
		self.use_residual = use_residual
		self.return_indices = return_indices
		self.activation = activation
  
		# 3 columns of 4 convolutional layers (32,k)
		self.inception_1 = Inception(
							in_channels=in_channels,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation, use_hybrid=use_hybrid,
							return_indices=return_indices
							)
		self.inception_2 = Inception(
							in_channels=4*n_filters if not use_hybrid else 4*n_filters+17,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)
		self.inception_3 = Inception(
							in_channels=4*n_filters,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)
  
		# 1 convolutional layer of n=128 filters of size k=1 
		if self.use_residual:
			self.residual = nn.Sequential(
								nn.Conv1d(
									in_channels=in_channels, 
									out_channels=4*n_filters,
									kernel_size=1,
									stride=1,
									padding=0
									),
								nn.BatchNorm1d(
									num_features=4*n_filters
									)
								)

	def forward(self, X):
		if self.return_indices:
			Z, i1 = self.inception_1(X)
			Z, i2 = self.inception_2(Z)
			Z, i3 = self.inception_3(Z)
		else:
			Z = self.inception_1(X)
			Z = self.inception_2(Z)
			Z = self.inception_3(Z)
		if self.use_residual:
			Z = Z + self.residual(X)
			Z = self.activation(Z)
		if self.return_indices:
			return Z,[i1, i2, i3]
		else:
			return Z


class InceptionTime(nn.Module):
    def __init__(self, in_channels, number_classes, kernel_sizes=[10,20,40], n_filters=32, bottleneck_channels=32, use_residual=True, activation=nn.ReLU()):
        super(InceptionTime, self).__init__()
        self.inception_block1 = InceptionBlock(
			in_channels=in_channels,
			n_filters=n_filters,
			kernel_sizes=kernel_sizes,
			bottleneck_channels=bottleneck_channels,
			use_residual=use_residual,
			activation=activation
        )
        self.inception_block2 = InceptionBlock(
			in_channels=n_filters*4,
			n_filters=n_filters,
			kernel_sizes=kernel_sizes,
			bottleneck_channels=bottleneck_channels,
			use_residual=use_residual,
			activation=activation
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = Flatten(out_features=n_filters*4*1)
        self.fc = nn.Linear(in_features=n_filters*4*1, out_features=number_classes)

    def forward(self, X):
        X = self.inception_block1(X)
        X = self.inception_block2(X)
        X = self.avg_pool(X)
        X = self.flatten(X)
        X = self.fc(X)
        return X


class DualInceptionTime(nn.Module):
    def __init__(self, in_channels_s2, in_channels_planet, number_classes, kernel_sizes_s2, kernel_sizes_planet, n_filters=32, bottleneck_channels=32, activation=nn.ReLU()):
        super(DualInceptionTime, self).__init__()
        self.inception_s2 = nn.Sequential(
            InceptionBlock(in_channels=in_channels_s2, 
                        kernel_sizes=kernel_sizes_s2, 
                        n_filters=n_filters, 
                        bottleneck_channels=bottleneck_channels, 
                        use_residual=True, 
                        activation=activation),
            InceptionBlock(in_channels=n_filters*4, 
                        kernel_sizes=kernel_sizes_s2, 
                        n_filters=n_filters, 
                        bottleneck_channels=bottleneck_channels, 
                        use_residual=True, 
                        activation=activation),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=n_filters*4*1)
        )
        self.inception_planet = nn.Sequential(
            InceptionBlock(in_channels=in_channels_planet, 
                        kernel_sizes=kernel_sizes_planet, 
                        n_filters=n_filters, 
                        bottleneck_channels=bottleneck_channels, 
                        use_residual=True, 
                        activation=activation),
            InceptionBlock(in_channels=n_filters*4, 
                        kernel_sizes=kernel_sizes_planet, 
                        n_filters=n_filters, 
                        bottleneck_channels=bottleneck_channels, 
                        use_residual=True, 
                        activation=activation),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=n_filters*4*1)
        )
        self.classifier = nn.Linear(in_features=2*n_filters*4*1, out_features=number_classes)
        
    def forward(self, x_s2, x_planet):
        x_s2 = self.inception_s2(x_s2)
        x_planet = self.inception_planet(x_planet)
        x = torch.cat((x_s2, x_planet), dim=1)
        x = self.classifier(x)
        return x


class HybridLayer(nn.Module):
    def __init__(self, input_channels, kernel_sizes=[2, 4, 8, 16, 32, 64], activation=nn.ReLU()):
        super(HybridLayer, self).__init__()
        self.input_channels = input_channels
        self.kernel_sizes = kernel_sizes
        self.conv_layers = self._create_conv_layers() 
        self.activation = activation  

    def _create_conv_layers(self):
        layers = []
        for kernel_size in self.kernel_sizes:
            # Increasing detection filter
            filter_inc = self._create_filter(kernel_size, pattern='increase')
            conv_inc = nn.Conv1d(in_channels=self.input_channels, out_channels=1, kernel_size=kernel_size,
                                 padding='same', bias=False)
            conv_inc.weight.data = filter_inc
            conv_inc.weight.requires_grad = False
            layers.append(conv_inc)

            # Decreasing detection filter
            filter_dec = self._create_filter(kernel_size, pattern='decrease')
            conv_dec = nn.Conv1d(in_channels=self.input_channels, out_channels=1, kernel_size=kernel_size,
                                 padding='same', bias=False)
            conv_dec.weight.data = filter_dec
            conv_dec.weight.requires_grad = False
            layers.append(conv_dec)

            # Peak detection filter, excluding the smallest kernel size for symmetry
            if kernel_size > 2: # because only 5 peak filters
                filter_peak = self._create_filter(kernel_size, pattern='peak')
                conv_peak = nn.Conv1d(in_channels=self.input_channels, out_channels=1, kernel_size=kernel_size + kernel_size // 2,
                                      padding='same', bias=False)
                conv_peak.weight.data = filter_peak
                conv_peak.weight.requires_grad = False                
                layers.append(conv_peak)
        return nn.ModuleList(layers)

    def _create_filter(self, k, pattern):
        if pattern == 'increase':
            filter_ = np.ones((1, self.input_channels, k)) # np.ones((row, column, depth)), each (row, column) has 'depth' values
            filter_[:, :, np.arange(k) % 2 == 0] = -1
        elif pattern == 'decrease':
            filter_ = np.ones((1, self.input_channels, k))
            filter_[:, :, np.arange(k) % 2 != 0] = -1
        elif pattern == 'peak':
            filter_ = np.zeros((1, self.input_channels, k+k//2))
            xmesh = np.linspace(start=0, stop=1, num=k // 4 + 1)[1:]
            filter_left = xmesh ** 2
            filter_right = np.flip(filter_left)
            filter_[:, :, 0:k//4] = -filter_left
            filter_[:, :, k//4:k//2] = -filter_right
            filter_[:, :, k//2:3*k//4] = 2 * filter_left
            filter_[:, :, 3*k//4:k] = 2 * filter_right
            filter_[:, :, k:5*k//4] = -filter_left
            filter_[:, :, 5*k//4:] = -filter_right
        return torch.tensor(filter_, dtype=torch.float32)

    def forward(self, x):
        outputs = []
        for conv in self.conv_layers:
            outputs.append(conv(x))
        outputs = torch.cat(outputs, dim=1)
        return self.activation(outputs)
    

class HybridInceptionTime(nn.Module):
    def __init__(self, in_channels, number_classes, kernel_sizes=[10, 20, 40], n_filters=32, bottleneck_channels=32):
        super(HybridInceptionTime, self).__init__()
        self.inception_block1 = InceptionBlock(
			in_channels=in_channels,
			n_filters=n_filters,
			kernel_sizes=kernel_sizes,
   			bottleneck_channels=bottleneck_channels,
			use_residual=True,
			activation=nn.ReLU(),
			use_hybrid=True
		)
        
        self.inception_block2 = InceptionBlock(
			in_channels=n_filters*4,
			n_filters=n_filters,
   			kernel_sizes=kernel_sizes,
			bottleneck_channels=bottleneck_channels,
			use_residual=True,
			activation=nn.ReLU(),
   			use_hybrid=False
		)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = Flatten(out_features=n_filters*4*1)
        self.fc = nn.Linear(in_features=n_filters*4*1, out_features=number_classes)
        
    def forward(self, X):
        X = self.inception_block1(X)
        X = self.inception_block2(X)
        X = self.avg_pool(X)
        X = self.flatten(X)
        X = self.fc(X)
        return X


class DualHybridInceptionTime(nn.Module):
    def __init__(self, in_channels_s2, in_channels_planet, number_classes, kernel_sizes_s2, kernel_sizes_planet, n_filters=32, bottleneck_channels=32):
        super(DualHybridInceptionTime, self).__init__()
        self.hinception_s2 = nn.Sequential(
            InceptionBlock(in_channels=in_channels_s2, 
                           kernel_sizes=kernel_sizes_s2, 
                           n_filters=n_filters,
                           bottleneck_channels=bottleneck_channels, 
                           use_residual=True, 
                           activation=nn.ReLU(), 
                           use_hybrid=True),
            InceptionBlock(in_channels=n_filters*4, 
                           kernel_sizes=kernel_sizes_s2, 
                           n_filters=n_filters,
                           bottleneck_channels=bottleneck_channels, 
                           use_residual=True, 
                           activation=nn.ReLU(), 
                           use_hybrid=False),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=n_filters*4*1)
        )
        self.hinception_planet = nn.Sequential(
            InceptionBlock(in_channels=in_channels_planet, 
                           kernel_sizes=kernel_sizes_planet, 
                           n_filters=n_filters,
                           bottleneck_channels=bottleneck_channels, 
                           use_residual=True, activation=nn.ReLU(), 
                           use_hybrid=True),
            InceptionBlock(in_channels=n_filters*4, 
                           kernel_sizes=kernel_sizes_planet, 
                           n_filters=n_filters,
                           bottleneck_channels=bottleneck_channels, 
                           use_residual=True, 
                           activation=nn.ReLU(), 
                           use_hybrid=False),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=n_filters*4*1)
        )
        self.classifier = nn.Linear(in_features=2*n_filters*4*1, out_features=number_classes)
    
    def forward(self, x_s2, x_planet):
        x_s2 = self.hinception_s2(x_s2)
        x_planet = self.hinception_planet(x_planet)
        x = torch.cat((x_s2, x_planet), dim=1)
        x = self.classifier(x)
        return x

"""
____________________________________________________________________________________
Credits:

The Transformer model has been inspired by the following paper:
'Attention Is All You Need': https://arxiv.org/abs/1706.03762

The implementation applied to time series data has been inspired by the following repository:
https://github.com/maxjcohen/transformer
____________________________________________________________________________________
"""

def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    

class Transformer(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension. number of bands
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    """
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 pos_enc = True,
                 decoding = False):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()
        self._d_model = d_model
        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout) for _ in range(N)])
        self._embedding = nn.Linear(d_input, d_model)
        self._output_layer = nn.Linear(d_model*params.temp_depth, d_output)
        self.pos_enc = pos_enc
        self.decoding = decoding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        x = x.permute(0, 2, 1)  # Original dataset size (batch_size, d_input, K)
        K = x.shape[1] # number of dates

        # Embedding module
        encoding = self._embedding(x) * math.sqrt(self._d_model) # size (batch_size, K, d_model)
        
        # Add position encoding
        if self.pos_enc:
            positional_encoding = generate_original_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding) # size encoding (batch_size, K, d_model) + positional_encoding(K, d_model) -> brodcasting

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)
            
        # Decoding stack
        if self.decoding:
            decoding = encoding
            if self.pos_enc:
                positional_encoding = generate_original_PE(K, self._d_model)
                positional_encoding = positional_encoding.to(decoding.device)
                decoding.add_(positional_encoding)
            for layer in self.layers_decoding:
                decoding = layer(decoding, encoding)
            encoding = decoding

        # Output
        output = encoding
        lengths = [K for X in range(x.shape[0])]
        padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=K).to(x.device)  # (batch_size, padded_length) boolean tensor, "1" means keep
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self._output_layer(output)  # (batch_size, num_classes)
        return output
    

class DualTransformer(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 pos_enc = True,
                 decoding = False):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._d_model = d_model
        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout) for _ in range(N)])
        self._embedding = nn.Linear(d_input, d_model)
        self._output_layer = nn.Linear(d_model*(params.temp_depth_Planet+params.temp_depth_S2), d_output)
        self.pos_enc = pos_enc
        self.decoding = decoding

        # Convolution 1D to reduce S2 dim to Planet dim
        self.conv1d = nn.Conv1d(in_channels=10, out_channels=4, kernel_size=1, bias=False)

    def forward(self, x_s2: torch.Tensor, x_planet: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        x_s2 = self.conv1d(x_s2)
        
        # Concatenate S2 and Planet data
        all_dates = sorted(params.list_dates_S2 + params.list_dates_Planet)
        date_to_index = {date: i for i, date in enumerate(all_dates)}

        # Initialise a final tensor with zeros of target size
        x = torch.zeros(x_planet.shape[0], 4, len(all_dates)).to(x_s2.device)

        # Insert S2 and Planet data into final tensor
        for i, date in enumerate(params.list_dates_S2):
            x[:, :, date_to_index[date]] = x_s2[:, :, i]
        for i, date in enumerate(params.list_dates_Planet):
            x[:, :, date_to_index[date]] = x_planet[:, :, i]
        
        x = x.permute(0, 2, 1)  # Original dataset size (batch_size, d_input, K)
        K = x.shape[1] # number of dates

        # Embedding module
        encoding = self._embedding(x) * math.sqrt(self._d_model) # size (batch_size, K, d_model)
        
        # Add position encoding
        if self.pos_enc:
            positional_encoding = generate_original_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding) # size encoding(batch_size, K, d_model) + positional_encoding(K, d_model) -> brodcasting

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Decoding stack
        if self.decoding:
            decoding = encoding
            if self.pos_enc:
                positional_encoding = generate_original_PE(K, self._d_model)
                positional_encoding = positional_encoding.to(decoding.device)
                decoding.add_(positional_encoding)
            for layer in self.layers_decoding:
                decoding = layer(decoding, encoding)
            encoding = decoding
        
        # Output
        output = encoding
        max_len = len(all_dates)
        lengths = [max_len for X in range(x.shape[0])] 
        padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len).to(x_s2.device)  # (batch_size, padded_length) boolean tensor, "1" means keep
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self._output_layer(output)  # (batch_size, num_classes)
        return output
    

"""
____________________________________________________________________________________
Credits:

The second transformer model has been inspired by the following paper:
'A Transformer-based Framework for Multivariate Time Series Representation Learning': https://arxiv.org/abs/2010.02803

The implementation of the second transformer model has been inspired by the following repository:
https://github.com/gzerveas/mvts_transformer
____________________________________________________________________________________
"""

class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)
        
    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU()):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = BatchNorm1d(d_model, eps=1e-5) # normalizes each feature across batch samples and time steps, is taking input size (batch_size, d_model, seq_length)
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = activation

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, is_causal: Optional[bool] = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required). // seq_length, batch_size, d_model
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """
    def __init__(self, feat_dim, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, activation=nn.GELU(), norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = FixedPositionalEncoding(d_model, dropout=dropout*(1.0 - freeze), max_len=params.temp_depth)
        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.dropout1 = nn.Dropout(dropout)
        self.act = activation
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = nn.Linear(d_model * params.temp_depth, num_classes)

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        inp = X.permute(0, 2, 1) # original data size (batch_size, feat_dim, seq_length)        
        K = X.shape[2] # number of dates
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = inp.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        lengths = [K for x in range(X.shape[0])] 
        padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=K).to(X.device)  # (batch_size, padded_length) boolean tensor, "1" means keep
        encoding = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        
        # Output
        output = encoding
        output = self.act(output) # ?
        output = output.permute(1, 0, 2)  # ? (batch_size, seq_length, d_model)
        output = self.dropout1(output) # ?
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)
        return output
    
    
class TSTransformerEncoderClassiregressorDual(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """
    def __init__(self, feat_dim_s2, feat_dim_planet, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, activation=nn.GELU(), norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassiregressorDual, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.project_inp_s2 = nn.Linear(feat_dim_s2, d_model)
        self.project_inp_planet = nn.Linear(feat_dim_planet, d_model)
        self.pos_enc = FixedPositionalEncoding(d_model, dropout=dropout*(1.0 - freeze), max_len=params.temp_depth_Planet)
        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.dropout1 = nn.Dropout(dropout)
        self.act = activation
        self.feat_dim_s2 = feat_dim_s2
        self.feat_dim_planet = feat_dim_planet
        self.num_classes = num_classes
        self.output_layer = nn.Linear(d_model*(params.temp_depth_S2+params.temp_depth_Planet), num_classes)
        # self.output_layer = nn.Linear(d_model*(347+221), num_classes)

    def forward(self, X_s2, X_planet):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        ### S2
        inp_s2 = X_s2.permute(0, 2, 1) # original data size (batch_size, feat_dim, seq_length)
        K = X_s2.shape[2] # number of dates
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp_s2 = inp_s2.permute(1, 0, 2)
        inp_s2 = self.project_inp_s2(inp_s2) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp_s2 = self.pos_enc(inp_s2)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        lengths_s2 = [K for x in range(X_s2.shape[0])] 
        padding_masks_s2 = padding_mask(torch.tensor(lengths_s2, dtype=torch.int16), max_len=K).to(X_s2.device)  # (batch_size, padded_length) boolean tensor, "1" means keep
        output_s2 = self.transformer_encoder(inp_s2, src_key_padding_mask=~padding_masks_s2)  # (seq_length, batch_size, d_model)
        output_s2 = self.act(output_s2)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output_s2 = output_s2.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output_s2 = self.dropout1(output_s2)
        output_s2 = output_s2 * padding_masks_s2.unsqueeze(-1)  # zero-out padding embeddings
        output_s2 = output_s2.reshape(output_s2.shape[0], -1)  # (batch_size, seq_length_s2 * d_model)
        
        ### Planet
        inp_planet = X_planet.permute(0, 2, 1) 
        K = X_planet.shape[2] # number of dates       
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp_planet = inp_planet.permute(1, 0, 2)
        inp_planet = self.project_inp_planet(inp_planet) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp_planet = self.pos_enc(inp_planet)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        lengths_planet = [K for x in range(X_planet.shape[0])] 
        padding_masks_planet = padding_mask(torch.tensor(lengths_planet, dtype=torch.int16), max_len=K).to(X_planet.device)  # (batch_size, padded_length) boolean tensor, "1" means keep
        output_planet = self.transformer_encoder(inp_planet, src_key_padding_mask=~padding_masks_planet)  # (seq_length, batch_size, d_model)
        output_planet = self.act(output_planet)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output_planet = output_planet.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output_planet = self.dropout1(output_planet)
        output_planet = output_planet * padding_masks_planet.unsqueeze(-1)  # zero-out padding embeddings
        output_planet = output_planet.reshape(output_planet.shape[0], -1)  # (batch_size, seq_length_ps * d_model)
        
        # Concatenate S2 and Planet data and output
        output = torch.cat((output_s2, output_planet), dim=1) # (batch_size, (seq_length_s2+seq_length_planet) * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)
        return output
# ____________________________________________________________________________________________

"""
____________________________________________________________________________________
Credits:

The LITE model has been inspired by the following paper:
'LITE: Light Inception with boosTing tEchniques for Time Series Classification': https://doi.org/10.1109/DSAA60987.2023.10302569

The implementation of the LITE model has been inspired by the following repository originally written in Keras:
https://github.com/MSD-IRIMAS/LITE/tree/main
____________________________________________________________________________________
"""

class LITEBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes = [10, 20, 40], dilation_rate = [1, 2, 4],
                 use_hybrid=True, use_dilation=True, use_multiplexing=True):
        super(LITEBlock, self).__init__()

        self.n_filters = n_filters
        self.use_dilation = use_dilation
        self.use_multiplexing = use_multiplexing        
        self.use_hybrid = use_hybrid
        
        if not use_multiplexing:
            n_filters = self.n_filters * 3
        else:
            n_filters = self.n_filters
            
        # the 3 convolutional layers with n_filters=32 of size kernel_sizes= 40, 20, 10
        self.conv_1 = nn.Conv1d(
                        in_channels=in_channels, 
                        out_channels=n_filters, 
                        kernel_size=kernel_sizes[0], 
                        stride=1, 
                        padding='same', 
                        dilation = dilation_rate[0], 
                        bias=False
                        )
        self.conv_2 = nn.Conv1d(
                        in_channels=in_channels, 
                        out_channels=n_filters, 
                        kernel_size=kernel_sizes[1], 
                        stride=1, 
                        padding='same',
                        dilation = dilation_rate[0], 
                        bias=False
                        )
        self.conv_3 = nn.Conv1d(
                        in_channels=in_channels, 
                        out_channels=n_filters, 
                        kernel_size=kernel_sizes[2], 
                        stride=1, 
                        padding='same', 
                        dilation = dilation_rate[0], 
                        bias=False
                        )
        if use_hybrid:
            self.hybrid = HybridLayer(input_channels=in_channels)
            self.batch_norm = nn.BatchNorm1d(num_features=3*n_filters+17)
        else:
            self.batch_norm = nn.BatchNorm1d(num_features=3*n_filters)
        self.activation = nn.ReLU()
        self.depthwise = nn.Conv1d(
                        in_channels=3*n_filters+17,
                        out_channels = n_filters, 
                        kernel_size = kernel_sizes[1], 
                        stride=1, 
                        padding='same', 
                        dilation=dilation_rate[1], 
                        bias=False)   
        self.pointwise = nn.Conv1d(
                        in_channels = n_filters, 
                        out_channels = n_filters, 
                        kernel_size=kernel_sizes[0], 
                        stride=1, 
                        padding='same', 
                        dilation=dilation_rate[2],
                        bias=False)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = Flatten(out_features=n_filters)
    
    def forward(self, X): 
        """
        X shape is (batch_size, in_channels, seq_length)
        """
        Z1 = self.conv_1(X)
        Z2 = self.conv_2(X)
        Z3 = self.conv_3(X)
        
        if self.use_hybrid:
            Z4 = self.hybrid(X)
            Z = torch.cat([Z1, Z2, Z3, Z4], dim=1) # 3*n_filters+17=113 as output
        else:
            Z = torch.cat([Z1, Z2, Z3], dim=1) # 3*n_filters as output
        Z = self.activation(self.batch_norm(Z))
        Z = self.depthwise(Z)
        Z = self.pointwise(Z)
        Z = self.avg_pool(Z)
        Z = self.flatten(Z)
        return Z
    

class LITE(nn.Module):
    def __init__(self, in_channels, number_classes, n_filters=32, kernel_sizes = [10, 20, 40], dilation_rate = [1, 2, 4],
                 use_hybrid=True, use_dilation=True, use_multiplexing=True):
        super(LITE, self).__init__()
        self.lite = nn.Sequential(
            LITEBlock(in_channels, 
                      n_filters, 
                      kernel_sizes, 
                      dilation_rate, 
                      use_hybrid, 
                      use_dilation, 
                      use_multiplexing)
            )
        self.classifier = nn.Linear(in_features=n_filters, out_features=number_classes)

    def forward(self, X):
        X = self.lite(X)
        X = self.classifier(X)
        return X


class DualLITE(nn.Module):
    def __init__(self, in_channels_s2, in_channels_planet, number_classes, kernel_sizes_s2, kernel_sizes_planet,
                 dilation_rate=[1, 2, 4], n_filters=32, use_hybrid=True, use_dilation=True, use_multiplexing=True):
        super(DualLITE, self).__init__()
        self.lite_s2 = nn.Sequential(
            LITEBlock(in_channels_s2, 
                      n_filters, 
                      kernel_sizes_s2, 
                      dilation_rate, 
                      use_hybrid, 
                      use_dilation, 
                      use_multiplexing)
            )
        self.lite_planet = nn.Sequential(
            LITEBlock(in_channels_planet, 
                      n_filters, 
                      kernel_sizes_planet, 
                      dilation_rate, 
                      use_hybrid, 
                      use_dilation, 
                      use_multiplexing)
            )
        self.classifier = nn.Linear(in_features=2*n_filters, out_features=number_classes)
        
    def forward(self, x_s2, x_planet):
        x_s2 = self.lite_s2(x_s2)
        x_planet = self.lite_planet(x_planet)
        x = torch.cat([x_s2, x_planet], dim=1)
        x = self.classifier(x)
        return x