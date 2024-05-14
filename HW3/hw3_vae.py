import sys
import argparse
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import matplotlib.image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from hw3_utils import array_to_image, concat_images, batch_indices, load_mnist
import random



# The "encoder" model q(z|x)
class Encoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Encoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units
        
        self.fc1 = nn.Linear(data_dimension, hidden_units)
        self.fc2_mu = nn.Linear(hidden_units, latent_dimension)
        self.fc2_sigma = nn.Linear(hidden_units, latent_dimension)

    def forward(self, x):
        # Input: x input image [batch_size x data_dimension]
        # Output: parameters of a diagonal gaussian 
        #   mean : [batch_size x latent_dimension]
        #   variance : [batch_size x latent_dimension]

        hidden = torch.tanh(self.fc1(x))
        mu = self.fc2_mu(hidden)
        log_sigma_square = self.fc2_sigma(hidden)
        sigma_square = torch.exp(log_sigma_square)  
        return mu, sigma_square


# "decoder" Model p(x|z)
class Decoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Decoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units

        # TODO: deine the parameters of the decoder
        # fc1: a fully connected layer with 500 hidden units. 
        # fc2: a fully connected layer with 500 hidden units. 
        self.fc1 = nn.Linear(latent_dimension, hidden_units, dtype = torch.float32)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_units, hidden_units, dtype = torch.float32)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_units, data_dimension, dtype = torch.float32)
        self.third = nn.Sigmoid()


    def forward(self, z):
        # input
        #   z: latent codes sampled from the encoder [batch_size x latent_dimension]
        # output 
        #   p: a tensor of the same size as the image indicating the probability of every pixel being 1 [batch_size x data_dimension]

        # TODO: implement the decoder here. The decoder is a multi-layer perceptron with two hidden layers. 
        # The first layer is followed by a tanh non-linearity and the second layer by a sigmoid.
        out = self.fc1(z)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)
        out = self.third(out)
        return out


# VAE model
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dimension = args.latent_dimension
        self.hidden_units =  args.hidden_units
        self.data_dimension = args.data_dimension
        self.resume_training = args.resume_training
        self.batch_size = args.batch_size
        self.num_epoches = args.num_epoches
        self.e_path = args.e_path
        self.d_path = args.d_path

        # load and pre-process the data
        N_data, self.train_images, self.train_labels, test_images, test_labels = load_mnist()

        # Instantiate the encoder and decoder models 
        self.encoder = Encoder(self.latent_dimension, self.hidden_units, self.data_dimension)
        self.decoder = Decoder(self.latent_dimension, self.hidden_units, self.data_dimension)

        # Load the trained model parameters
        if self.resume_training:
            self.encoder.load_state_dict(torch.load(self.e_path))
            self.decoder.load_state_dict(torch.load(self.d_path))

    # Sample from Diagonal Gaussian z~N(μ,σ^2 I) 
    @staticmethod
    def sample_diagonal_gaussian(mu, sigma_square):
        # Inputs:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   sample: from a diagonal gaussian with mean mu and variance sigma_square [batch_size x latent_dimension]

        # TODO: Implement the reparameterization trick and return the sample z [batch_size x latent_dimension]
        sample = torch.randn_like(sigma_square)
        return mu + sample * torch.sqrt(sigma_square)

    # Sampler from Bernoulli
    @staticmethod
    def sample_Bernoulli(p):
        # Input: 
        #   p: the probability of pixels labeled 1 [batch_size x data_dimension]
        # Output:
        #   x: pixels'labels [batch_size x data_dimension], type should be torch.float32

        # TODO: Implement a sampler from a Bernoulli distribution
        # x = np.random.binomial(1, p.detach().numpy())
        
        # return torch.from_numpy(x).to(torch.float32)
        return torch.bernoulli(p)


    # Compute Log-pdf of z under Diagonal Gaussian N(z|μ,σ^2 I)
    @staticmethod
    def logpdf_diagonal_gaussian(z, mu, sigma_square):
        # Input:
        #   z: sample [batch_size x latent_dimension]
        #   mu: mean of the gaussian distribution [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian distribution [batch_size x latent_dimension]
        # Output:
        #    logprob: log-probability of a diagomnal gaussian [batch_size]
        
        # TODO: implement the logpdf of a gaussian with mean mu and variance sigma_square*I
        logprob = -0.5 * torch.sum(np.log(2 * torch.pi) + torch.log(sigma_square) + (z - mu)**2 / sigma_square, dim=1)
        return logprob

    # Compute log-pdf of x under Bernoulli 
    @staticmethod
    def logpdf_bernoulli(x, p):
        # Input:
        #   x: samples [batch_size x data_dimension]
        #   p: the probability of the x being labeled 1 (p is the output of the decoder) [batch_size x data_dimension]
        # Output:
        #   logprob: log-probability of a bernoulli distribution [batch_size]

        # TODO: implement the log likelihood of a bernoulli distribution p(x)
        logprob = torch.sum(x * torch.log(p) + (1-x) * torch.log(1-p), dim = 1)
        return logprob
    
    # Sample z ~ q(z|x)
    def sample_z(self, mu, sigma_square):
        # input:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   zs: samples from q(z|x) [batch_size x latent_dimension] 
        
        zs = self.sample_diagonal_gaussian(mu, sigma_square)
        return zs 


    # Variational Objective
    def elbo_loss(self, sampled_z, mu, sigma_square, x, p):
        # Inputs
        #   sampled_z: samples z from the encoder [batch_size x latent_dimension]
        #   mu:
        #   sigma_square: parameters of q(z|x) [batch_size x latent_dimension]
        #   x: data samples [batch_size x data_dimension]
        #   p: the probability of a pixel being labeled 1 [batch_size x data_dimension]
        # Output
        #   elbo: the ELBO loss (scalar)

        # log_q(z|x) logprobability of z under approximate posterior N(μ,σ)
        log_q = self.logpdf_diagonal_gaussian(sampled_z, mu, sigma_square)
        
        # log_p_z(z) log probability of z under prior
        z_mu = torch.FloatTensor([0]*self.latent_dimension).repeat(sampled_z.shape[0], 1)
        z_sigma = torch.FloatTensor([1]*self.latent_dimension).repeat(sampled_z.shape[0], 1)
        log_p_z = self.logpdf_diagonal_gaussian(sampled_z, z_mu, z_sigma)

        # log_p(x|z) - conditional probability of data given latents.
        log_p = self.logpdf_bernoulli(x, p)
        

        # TODO: implement the ELBO loss using log_q, log_p_z and log_p
        # elbo = torch.sum(torch.exp(log_q) * (log_p_z - log_q + log_p))
        elbo = torch.mean((log_p_z - log_q + log_p))
        return elbo


    def train(self):
        
        # Set-up ADAM optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        adam_optimizer = optim.Adam(params)

        # Train for ~200 epochs 
        num_batches = int(np.ceil(len(self.train_images) / self.batch_size))
        num_iters = self.num_epoches * num_batches
        
        for i in range(num_iters):
            x_minibatch = self.train_images[batch_indices(i, num_batches, self.batch_size),:]
            adam_optimizer.zero_grad()

            mu, sigma_square = self.encoder(x_minibatch)
            zs = self.sample_z(mu, sigma_square)
            p = self.decoder(zs)
            elbo = self.elbo_loss(zs, mu, sigma_square, x_minibatch, p)
            total_loss = -elbo
            total_loss.backward()
            adam_optimizer.step()
            if i%100 == 0:
                print("Epoch: " + str(i//num_batches) + ", Iter: " + str(i) + ", ELBO:" + str(elbo.item()))

        # Save Optimized Model Parameters
        torch.save(self.encoder.state_dict(), self.e_path)
        torch.save(self.decoder.state_dict(), self.d_path)


    # Generate digits using the VAE
    def visualize_data_space(self):
        # TODO: Sample 10 z from prior
        zs = torch.randn(10, self.latent_dimension)
        
        # TODO: For each z, plot p(x|z)
        # TODO: Sample x from p(x|z) 
        # TODO: Concatenate plots into a figure (use the function concat_images)
        
        images = np.zeros((20,28,28))

        for i, z in enumerate(zs):
            px = self.decoder(z)
            xs = self.sample_Bernoulli(px)
            
            image_px = array_to_image(px.detach().numpy())
            image_xs = array_to_image(xs.detach().numpy())
            
            
            images[i] = image_px
            images[i + 10] = image_xs
        
        plt.imshow(concat_images(images, 10, 2), cmap='gray')

        plt.show()
        # TODO: Save the generated figure and include it in your report
        return
        
    # Produce a scatter plot in the latent space, where each point in the plot will be the mean vector 
    # for the distribution $q(z|x)$ given by the encoder. Further, we will colour each point in the plot 
    # by the class label for the input data. Each point in the plot is colored by the class label for 
    # the input data.
    # The latent space should have learned to distinguish between elements from different classes, even though 
    # we never provided class labels to the model!
    def visualize_latent_space(self):
        
        # TODO: Encode the training data self.train_images
        mu, sigma_square = self.encoder(self.train_images)
        

        # TODO: Take the mean vector of each encoding
        npmu = mu.detach().numpy()

        # TODO: Plot these mean vectors in the latent space with a scatter
        labels =  (self.train_labels @ (torch.tensor([0,1,2,3,4,5,6,7,8,9], dtype = torch.float32).reshape(10,1))).reshape(self.train_labels.shape[0])
    
        # Colour each point depending on the class label 
        colors = ["red","green","black","yellow","blue", "cyan", "magenta", (0.32,0.53,0.7), (0.72,0.23,0.41), (0.56,0.43,0.11)]
        
        for i in range(labels.shape[0]):
            plt.plot(mu[i][0].item(), mu[i][1].item(), marker='.', markersize=3, color = colors[int(labels[i].item())])
        
        # TODO: Save the generated figure and include it in your report
        plt.show()
        return


    # Function which gives linear interpolation z_α between za and zb
    @staticmethod
    def interpolate_mu(mua, mub, alpha = 0.5):
        return alpha*mua + (1-alpha)*mub


    # A common technique to assess latent representations is to interpolate between two points.
    # Here we will encode 3 pairs of data points with different classes.
    # Then we will linearly interpolate between the mean vectors of their encodings. 
    # We will plot the generative distributions along the linear interpolation.
    def visualize_inter_class_interpolation(self):
        
        # TODO: Sample 3 pairs of data with different classes
        labels =  (self.train_labels @ (torch.tensor([0,1,2,3,4,5,6,7,8,9], dtype = torch.float32).reshape(10,1))).reshape(self.train_labels.shape[0])
        pairs = []
        count = 0
        for i in range(labels.shape[0]):
            if random.randint(1, 700) > 1:
                continue
            for j in range(i+1, labels.shape[0]):
                if random.randint(1, 700) > 1:
                    continue
                if labels[i].item() != labels[j].item():
                    pairs.append((i,j))
                    count+=1
                    break   
            if count == 3:
                break
        
        # TODO: Encode the data in each pair, and take the mean vectors
        # TODO: Linearly interpolate between these mean vectors (Use the function interpolate_mu)
        # TODO: Along the interpolation, plot the distributions p(x|z_α)

        images = np.zeros((3,28,28))
        for i in range(3):
            p = pairs[i]
            mu1, sq1 = self.encoder(self.train_images[p[0]])
            mu2, sq2 = self.encoder(self.train_images[p[1]])
            z_alpha = self.interpolate_mu(mu1,mu2,0.1)
            px = self.decoder(z_alpha)
            image_px = array_to_image(px.detach().numpy())
            images[i] = image_px

        
        plt.imshow(concat_images(images, 1, 3), cmap='gray')
        plt.show()

        # Concatenate these plots into one figure
        return
        
      

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--e_path', type=str, default="./e_params.pkl", help='Path to the encoder parameters.')
    parser.add_argument('--d_path', type=str, default="./d_params.pkl", help='Path to the decoder parameters.')
    parser.add_argument('--hidden_units', type=int, default=500, help='Number of hidden units of the encoder and decoder models.')
    parser.add_argument('--latent_dimension', type=int, default='2', help='Dimensionality of the latent space.')
    parser.add_argument('--data_dimension', type=int, default='784', help='Dimensionality of the data space.')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--num_epoches', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
    
    args = parser.parse_args()

    # args = easydict.EasyDict({
    #     "e_path": "./e_params.pkl",
    #     "d_path": "./d_params.pkl",
    #     "hidden_units": 500,
    #     "latent_dimension": 2,
    #     "data_dimension": 784,
    #     "resume_training": True,
    #     "seed": 1234,
    #     "num_epoches": 200,
    #     "batch_size": 100
    # })
    return args


def main():
    # read the function arguments
    args = parse_args()

    # set the random seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # train the model 
    vae = VAE(args)
    vae.train()

    # visualize the latent space
    vae.visualize_data_space()
    vae.visualize_latent_space()
    vae.visualize_inter_class_interpolation()

main()
