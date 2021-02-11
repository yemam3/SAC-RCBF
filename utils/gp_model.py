""" Adapted almost directly from:
https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/Simple_GP_Regression_CUDA.html

Training is performed rapidly (and exactly) using GPUs and prediction is done very rapidly using LOVE.

TODO: GPyTorch can kind of support online learning if needed... https://github.com/cornellius-gp/gpytorch/issues/1200
"""

import torch
import gpytorch

from utils.util import to_tensor, to_numpy


class BaseGPy(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GPyDisturbanceEstimator:
    """
    A wrapper around teh BaseGPy model above.
    """

    def __init__(self, train_x, train_y, likelihood=None):

        if not torch.is_tensor(train_x):
            train_x = to_tensor(train_x)
        if not torch.is_tensor(train_y):
            train_y = to_tensor(train_y)
        self.train_x = train_x
        self.train_y = train_y
        if not likelihood:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood = likelihood

        self.model = BaseGPy(train_x, train_y, likelihood)

        if torch.cuda.is_available():
            self.cuda()

    def cuda(self):

        self.train_x = self.train_x.cuda()
        self.train_y = self.train_y.cuda()
        self.model = self.model.cuda()
        self.likelihood = self.likelihood.cuda()

    def train(self, training_iter, verbose=False):

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            if verbose:
                print('\tIter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    self.model.covar_module.base_kernel.lengthscale.item(),
                    self.model.likelihood.noise.item()
                ))
            optimizer.step()

    def predict(self, test_x):

        # Convert to torch tensor
        is_tensor = torch.is_tensor(test_x)
        if not is_tensor:
           test_x = to_tensor(test_x)
        # Move to GPU
        if torch.cuda.is_available():
            test_x = test_x.cuda()

        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
            pred_dict = dict()
            pred_dict['mean'] = observed_pred.mean.cpu()
            pred_dict['f_var'] = observed_pred.variance.cpu()
            pred_dict['f_covar'] = observed_pred.covariance_matrix.cpu()
            lower_ci, upper_ci = observed_pred.confidence_region()
            pred_dict['lower_ci'] = lower_ci.cpu()
            pred_dict['upper_ci'] = upper_ci.cpu()

        # If they gave us ndarray, we give back ndarray
        if not is_tensor:
            for key, val in pred_dict.items():
                pred_dict[key] = to_numpy(val)

        return pred_dict


if __name__ == '__main__':

    # this is for running the notebook in our testing framework
    import os
    import matplotlib.pyplot as plt
    import math

    smoke_test = ('CI' in os.environ)
    training_iter = 2 if smoke_test else 50
    # Training data is 11 points in [0,1] inclusive regularly spaced
    train_x = torch.linspace(0, 1, 100)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2

    disturb_estimator = GPyDisturbanceEstimator(train_x, train_y)
    disturb_estimator.train(training_iter)

    print('Now testing model!!')
    test_x = torch.linspace(0, 1, 51)
    prediction = disturb_estimator.predict(test_x)
    train_x = train_x.cpu()
    train_y = train_y.cpu()
    test_x = test_x.cpu()

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), prediction['mean'].numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), prediction['lower_ci'].numpy(), prediction['upper_ci'].numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()