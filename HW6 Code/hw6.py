
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
import hw6_utils as utils
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConvNet(nn.Module):
    """
    A model with one 1d conv layer and one 1d transpose conv layer.
    It generates the embeddings with the conv layer with the given 
    kernel size and the number of output channels. Then uses ReLU 
    activation on the generated embedding and passes it to a 
    transpose conv layer that has the same kernel size as the conv
    layer and one output channel.

    Arguments:
        kernel_size: size of the kernel for 1d convolution.
        length: length of the sequence (10 in the given data)
        out_chan: number of output channels for the conv layer
            and number of input layers in the trasnpose conv layer.
        bias: binary flag for using bias.

    Returns: 
        the predicted mapping (size: n x length)
    """
    def __init__(self, kernel_size=3, length=10, out_chan=32, bias=True):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv1d(1, out_chan, kernel_size, bias = bias)
        self.relu = nn.ReLU()
        self.transpose_conv = nn.ConvTranspose1d(out_chan, 1, kernel_size, bias=bias)
        self.length = length

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.transpose_conv(x)
        x = x.squeeze(1)
        return x


class Attention(nn.Module):
    """
    An attention-based model with one single head. It uses linear layer
    without bias to generate query key and value for the embedding of 
    each element of the input vector. 

    Arguments:
        length: length of the sequence (10 in the given data)
        embedding_dim: the embedding dimension for each element
            of the sequence.
        positional_encoding: a booliean flag which turns on the 
            positional encoding when set to True.

    Returns: 
        the predicted mapping (size: n x length)
    """
    def __init__(self, length=10, embedding_dim=16, positional_encoding=True):
        super().__init__()

        self.embedding = nn.Embedding(2, embedding_dim)

        # TODO: Add 3 linear layers with no bias for generating 
        # the query, key, and values
        self.q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.out = nn.Linear(embedding_dim, 1)

        self.attention = np.zeros((length,length))
        self.pos_encode = utils.PositionalEncoding(d_model=embedding_dim, max_len=length)
        self.positional_encoding = positional_encoding

    def compute_new_values(self, q, k, v):
        """
        Computes the attention matrix and the new values:
        
        Arguments:
            q: query
            k: key
            v: value

        Returns:
            values: the new values computed using the 
                attention matrix.
            attentions: attention matrix
        """
        # TODO: implement the function.
        score = torch.matmul(q, k.transpose(-2, -1) / np.sqrt(q.size(-1)))
        attentions = F.softmax(score, dim = -1)
        values = torch.matmul(attentions, v)
        return values, attentions.detach()

    def attention_mat(self):
        return np.mean(self.attention, axis=0)

    def forward(self, x):
        x = self.embedding(x.long())

        if self.positional_encoding:
            x_ = x.permute(1, 0, 2)
            x_ = self.pos_encode(x_)
            x = x_.permute(1, 0, 2)

        # TODO: compute the query, key, and value representations.
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        
        values, attention = self.compute_new_values(query, key, value)
        self.attention = attention.numpy()
        values = self.out(values)
        return values.view(x.shape[0],-1)


def train(model, epoch, optimizer, criterion, trainloader, log=True):
    model.train()
    train_loss = 0.0
    total_seen = 0
    correct = 0.0
    for batch_idx, inputs in enumerate(trainloader):
        inputs = inputs.float().to(device)
        seq_len = inputs.shape[-1]//2
        X = inputs[:,:seq_len]
        Y = inputs[:,seq_len:]

        model.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        predictions = torch.clip(outputs,0,1)
        predictions = (predictions>0.5).float()
        total_seen += Y.size(0)
        train_loss += loss.item()
        correct += predictions.eq(Y).sum().item()

    accuracy = 100.*correct/(seq_len*total_seen) 
    if log:
        print('Epoch: %d  Train Loss: %.3f | Train Acc: %.3f' % (epoch, train_loss/(batch_idx+1), accuracy))

    return accuracy


def test(model, testloader, log=True):
    model.eval()
    correct = 0.0
    predictions = None
    for batch_idx, inputs in enumerate(testloader):
        inputs = inputs.float().to(device)
        seq_len = inputs.shape[-1]//2
        X = inputs[:,:seq_len]
        Y = inputs[:,seq_len:]
        outputs = model(X)
        predictions = torch.clip(outputs,0,1)
        predictions = (predictions>0.5).float()
        correct += torch.prod(predictions.eq(Y).float(), dim=1).item()

    if log:
        print('Test Acc: %.3f' % (100.*correct/len(testloader)))
    return outputs.detach(), correct



if __name__ == "__main__":
    seed_val = 1
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    length = 10
    kernel_size = 3
    X = utils.load_data()
    n = X.shape[0]
    model_type = 'Attention'

    train_X = X[:int(0.8*n)]
    test_X = X[int(0.8*n):]

    trainloader = torch.utils.data.DataLoader(train_X, shuffle=True, batch_size=64, num_workers=1)
    testloader = torch.utils.data.DataLoader(test_X, shuffle=False, batch_size=1, num_workers=1)

    if model_type == 'Convolution':
        model = ConvNet(length=length, kernel_size=kernel_size).to(device) 
    elif model_type == 'Attention':
        model = Attention(length=length, positional_encoding=True).to(device) 
    else:
        print('not a valid model!')
        exit(0)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss(reduction="none")

    for epoch in range(21):
        train(model, epoch, optimizer, criterion, trainloader)
        if epoch % 10 == 0 and epoch > 0:
            test(model, testloader)
        plt.matshow(model.attention_mat())