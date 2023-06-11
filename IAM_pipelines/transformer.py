import numpy as np

import torch
import torch.nn as nn

import torchvision as tv

import torch.nn.functional as F

device = "cpu"

# resizes to largest width in batch x 128, keeping aspect ratio and padding image
class resizeImage(object):
    def __init__(self, resize_width, resize_height):
        self.resize_width = resize_width
        self.resize_height = resize_height
    
    def __call__(self, image):
        # check if resizing to correct height while keeping aspect ratio does not overshoot correct width
        aspect_ratio_width = int((self.resize_height / image.size(1)) * image.size(2))
        if (aspect_ratio_width > self.resize_width):
            # calculate max ratio of change for not overshooting resize width while keeping aspect ratio 
            max_ratio = self.resize_width / image.size(2)
            max_resize_height = int(max_ratio * image.size(1))
            # calc up and down padding
            padding_up = int(((self.resize_height - max_resize_height) / 2))
            padding_down = self.resize_height - max_resize_height - padding_up
            # change resize height to max calculated resize height
            new_resize_height = max_resize_height
        else:
            padding_up = 0
            padding_down = 0
            new_resize_height = self.resize_height

        # resize to correct image height, while keeping aspect ratio
        resize_transform = tv.transforms.Resize((new_resize_height, self.resize_width), antialias = True)
        resized = resize_transform(image)
        
        # pad to correct width (and height if necessary)
        padding_left = int(((self.resize_width - resized.size(2)) / 2))
        padding_right = self.resize_width - resized.size(2) - padding_left
        resized_padded = F.pad(resized, (padding_left, padding_right, padding_up, padding_down), mode = "constant", value = 255)

        return resized_padded

# paper adds gaussian noise to image (sqrt(0.1) * rand from gaussian distri)
class addGaussianNoise(object):
    def __call__(self, image):
        image += ((0.1**0.5) * torch.randn(image.shape)) * 0.75

        return image

class SinPosEncoding(nn.Module):
    def __init__(self, dimensionality):
        super(SinPosEncoding, self).__init__()
        self.dimensionality = dimensionality
        self.max_len = 1000

        # initialize encodings
        pos_encodings = torch.zeros((self.max_len, 1, self.dimensionality), dtype = torch.float32)

        # loop through size of matrix
        for j in range(self.max_len):
            for i in np.arange(int(self.dimensionality / 2)):
                # calc sin and cos wave offsets (alternated) for each row of matrix
                pos_encodings[j, 0, 2*i] = np.sin(j / (self.max_len ** ((2*i) / self.dimensionality)))
                pos_encodings[j, 0, 2*i + 1] = np.cos(j / (self.max_len ** ((2*i) / self.dimensionality)))
              
        # add to buffer for performance (?)
        self.register_buffer('pos_encodings', pos_encodings)

    def forward(self, input: torch.Tensor):
        # adds the positional encoding elementwise to the tensor (seqlength, batch, embeddims)
        input += self.pos_encodings[0:input.size(0)]

        return input

class CNN(nn.Module):
    def __init__(self, input_height, input_width):
        super(CNN, self).__init__()
        # convolutional block (5 convolutions)
        # first convolution
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = (3,3))
        width = input_width - 2
        height = input_height - 2
        self.leakyRelu = nn.LeakyReLU()     # reuse in later layers
        self.maxPool = nn.MaxPool2d((2,2))  # reuse in later layers
        width = int(np.floor(width/2))
        height = int(np.floor(height/2))
        self.layerNorm1 = nn.LayerNorm(normalized_shape = [8, height, width])
        self.dropout = nn.Dropout(0.2)      # reuse in later layers

        # second convolutional layer
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3, 3))
        width -= 2
        height -= 2
        # after maxpool
        width = int(np.floor(width/2))
        height = int(np.floor(height/2))
        self.layerNorm2 = nn.LayerNorm(normalized_shape = [16, height, width])

        # third convolutional layer
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 3))
        width -= 2
        height -= 2
        # after maxpool
        width = int(np.floor(width/2))
        height = int(np.floor(height/2))
        self.layerNorm3 = nn.LayerNorm(normalized_shape = [32, height, width])

        # forth convolutional layer
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3))
        width -= 2
        height -= 2
        # no maxpool
        self.layerNorm4 = nn.LayerNorm(normalized_shape = [64, height, width])

        # fifth convolutional layer (kernel size to better match shape of character)
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (4, 2))
        width -= 1
        height -= 3
        # no maxpool
        self.layerNorm5 = nn.LayerNorm(normalized_shape = [128, height, width])

        # following is convolution with width 1 which is used to flatten the current output
        self.flattenConv = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (height, 1))
        self.layerNorm6 = nn.LayerNorm(normalized_shape = [128, 1, width])

        # dense layer to upscale from 128 to 256
        self.dense = nn.Linear(in_features = 128, out_features = 256)

    def forward(self, input_img):
        # first conv
        conv_out = self.layerNorm1(self.maxPool(self.leakyRelu(self.conv1(input_img))))
        conv_out = self.dropout(conv_out)
        # second conv
        conv_out = self.layerNorm2(self.maxPool(self.leakyRelu(self.conv2(conv_out))))
        conv_out = self.dropout(conv_out)
        # third conv
        conv_out = self.layerNorm3(self.maxPool(self.leakyRelu(self.conv3(conv_out))))
        conv_out = self.dropout(conv_out)
        # forth conv
        conv_out = self.layerNorm4(self.leakyRelu(self.conv4(conv_out)))
        # fifth conv
        conv_out = self.layerNorm5(self.leakyRelu(self.conv5(conv_out)))

        # flatten layer
        conv_out = self.layerNorm6(self.leakyRelu(self.flattenConv(conv_out)))

        # reshape from ((batch, 128, 1, x) -> (batch, x, 1, 128)) for dense layer
        conv_out = torch.permute(conv_out, (0, 3, 2, 1))

        # upscale from 128 to 256
        conv_out = self.dense(conv_out)

        # reshape to (seq, batch, embed_dim)
        conv_out = torch.permute(conv_out.squeeze(2), (1, 0, 2))
        
        return conv_out

class HWRTransformerEncoder(nn.Module):
    def __init__(self, total_nr_of_tokens):
        super(HWRTransformerEncoder, self).__init__()
        # transformer encoder layers (4 stacked transformer encoder layers (4 headed attention))
        self.trans_encoder1 = nn.TransformerEncoderLayer(d_model = 256, nhead = 4, dim_feedforward = 1024, dropout = 0.2)
        self.trans_encoder2 = nn.TransformerEncoderLayer(d_model = 256, nhead = 4, dim_feedforward = 1024, dropout = 0.2)
        self.trans_encoder3 = nn.TransformerEncoderLayer(d_model = 256, nhead = 4, dim_feedforward = 1024, dropout = 0.2)
        self.trans_encoder4 = nn.TransformerEncoderLayer(d_model = 256, nhead = 4, dim_feedforward = 1024, dropout = 0.2)

        # dense layer for backprop CTC Loss of intermediate encoder result
        self.encoder_out_dense = nn.Linear(256, total_nr_of_tokens)

    def forward(self, encoder_input):
        # transformer encoder layers
        encoder_out = self.trans_encoder1(encoder_input)
        encoder_out = self.trans_encoder2(encoder_out)
        encoder_out = self.trans_encoder3(encoder_out)
        encoder_out = self.trans_encoder4(encoder_out)

        return encoder_out 

class HWRTransformerDecoder(nn.Module):
    def __init__(self, total_nr_of_tokens):
        super(HWRTransformerDecoder, self).__init__()

        # batch first = false -> input should be [target_len, batch_size, nr_tokens]
        # transformer decoder layers (4 stacked transformer encoder layers (4 headed attention))
        self.trans_decoder1 = nn.TransformerDecoderLayer(d_model = 256, nhead = 4, dim_feedforward = 1024, dropout = 0.2)
        self.trans_decoder2 = nn.TransformerDecoderLayer(d_model = 256, nhead = 4, dim_feedforward = 1024, dropout = 0.2)
        self.trans_decoder3 = nn.TransformerDecoderLayer(d_model = 256, nhead = 4, dim_feedforward = 1024, dropout = 0.2)
        self.trans_decoder4 = nn.TransformerDecoderLayer(d_model = 256, nhead = 4, dim_feedforward = 1024, dropout = 0.2)

        # reshape output to number of possible tokens
        self.decoder_out_dense = nn.Linear(256, total_nr_of_tokens)

    def forward(self, decoder_in, encoder_out, target_mask):
        # input encoder output and predicted chars into decoder
        decoder_out = self.trans_decoder1(decoder_in, encoder_out, target_mask)
        decoder_out = self.trans_decoder2(decoder_out, encoder_out, target_mask)
        decoder_out = self.trans_decoder3(decoder_out, encoder_out, target_mask)
        decoder_out = self.trans_decoder4(decoder_out, encoder_out, target_mask)

        # dense layer after decoder to predict one of all tokens (CE Loss)
        decoder_out = self.decoder_out_dense(decoder_out)

        return decoder_out

class HWRTransformer(nn.Module):
    def __init__(self, input_height, input_width, total_nr_of_tokens, longest_label_size):
        super(HWRTransformer, self).__init__()
        # CNN backbone to extract optical features of input image
        self.cnn = CNN(input_height, input_width)

        # pre-encoder positional information
        self.pos_encoding = SinPosEncoding(dimensionality = 256)

        # Transformer encoder
        self.transformer_encoder = HWRTransformerEncoder(total_nr_of_tokens)
        # dense layer and logsoftmax for intermediate output (to backprop with CTC Loss)
        self.encoder_out_dense = nn.Linear(256, total_nr_of_tokens)
        self.encoder_out_logsoftmax = nn.LogSoftmax(dim = 2)

        # character embedding (dim rule of thumb -> 4th sqrt of nr_embeddings: for ~80 = 3) 
        #      NOTE: wrong, appearantly dims (encoder output, target embedding) need to be the same
        # <PAD> embedding idx = 0
        self.char_embedding = nn.Embedding(total_nr_of_tokens, 256, padding_idx = 0)

        # Transformer decoder 
        self.decoder_target_mask = self.make_target_mask(longest_label_size)
        self.transformer_decoder = HWRTransformerDecoder(total_nr_of_tokens)

    # create a target mask for decoder input
    #   masks the future target characters from being seen by the model before they should
    def make_target_mask(self, size):
        mask = torch.zeros((size, size), dtype = torch.float32).to(device)
        
        for i in range(size):
            for j in range(size):
                if (j > i):
                    mask[i][j] = float('-inf')
        # print(mask)
        return mask

    def forward(self, input_image, decoder_in_embed_idxs):
        # forward through backbone convolutional neural network
        cnn_out = self.cnn(input_image)

        # add pre-encoder positional information
        cnn_out = self.pos_encoding(cnn_out)
        
        # forward through transformer encoder
        encoder_out = self.transformer_encoder(cnn_out)

        # print("Encoder out shape:", encoder_out.shape)

        # dense layer for intermediate output (to backprop with CTC Loss).
        # clone, otherwise inplace operation error (compute graph messes up)
        cloned_encoder_out = torch.clone(encoder_out)
        interm_encoder_out = self.encoder_out_logsoftmax(self.encoder_out_dense(cloned_encoder_out))

        # add pre-decoder positional information
        encoder_out = self.pos_encoding(encoder_out)

        # embed character indices for input into decoder
        shifted_target = self.char_embedding(decoder_in_embed_idxs)

        # reshape from (batch, seq_len, embed_dim) -> (seq_len, batch, embed_dim)
        shifted_target = torch.permute(shifted_target, (1, 0, 2))

        # add positional encoding to shifted targets
        shifted_target = self.pos_encoding(shifted_target)

        # forward through transformer decoder
        decoder_out = self.transformer_decoder(shifted_target, encoder_out, self.decoder_target_mask)

        return interm_encoder_out, decoder_out

class HybridLoss(nn.Module):
    def __init__(self, balance, nr_of_classes):
        super(HybridLoss, self).__init__()
        # allignment probabilities over encoded input sequence and target sequence
        #   labels are in conform with char to idx mapping (nr of classes - 1 indices)
        #   so I decided to make the blank index the next unused idx mapping
        self.interm_CTCloss = nn.CTCLoss()
        # difference between decodeced input sequence and target sequence
        self.output_CELoss = nn.CrossEntropyLoss()

        # balance between CTC Loss and CE Loss (R: [0, 1])
        if balance < 0 or balance > 1:
            raise ValueError("Balance should be a value between 0 (only output CELoss) and 1 (only intermediate CTCLoss)")
        self.balance = balance

    def setBalance(self, balance):
        self.balance = balance

    def forward(self, encoder_outputs, encoder_targets, label_lengths, decoder_outputs, decoder_targets):
        # create input lengths tensor based on the size of the encoder output 
        input_lengths = torch.tensor([encoder_outputs.size(0)]).repeat(encoder_outputs.size(1))

        # cross entropy needs (batch_size, seq_len, nr_tokens), right now its (seq_len, batch_size, nr_tokens)
        decoder_outputs = torch.permute(decoder_outputs, (1, 0, 2))
        decoder_targets = torch.permute(decoder_targets, (1, 0, 2))

        interm_loss = self.interm_CTCloss(encoder_outputs, encoder_targets, input_lengths, label_lengths)
        output_loss = self.output_CELoss(decoder_outputs, decoder_targets)

        return (self.balance * interm_loss + (1 - self.balance) * output_loss)