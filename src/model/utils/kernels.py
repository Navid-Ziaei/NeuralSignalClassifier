import torch
import torch.nn as nn

class ARDRBFKernel(nn.Module):
    """
    Implements the Automatic Relevance Determination (ARD) Radial Basis Function (RBF) kernel, also known as
    the ARD squared exponential kernel. This kernel function introduces a separate length scale parameter for
    each dimension of the input space, allowing it to adapt to different scales in different input dimensions.

    The ARD RBF kernel is defined as:

    k(x, x') = σ^2 * exp(-0.5 * Σ(α_q * (x_q - x'_q)^2))

    where σ^2 is the variance, α_q are the inverse square lengthscales, and x_q represents the q-th dimension of
    the input vectors x and x'.

    Attributes:
        input_dim (int): The number of dimensions of the input space.
        variance (nn.Parameter): The variance parameter σ^2 of the kernel.
        lengthscale (nn.Parameter): The length scale parameters α_q of the kernel, one for each input dimension.
    """

    def __init__(self, input_dim, variance=None, alpha=None, requires_grad=True):
        """
        Initializes the ARDRBFKernel class.

        Args:
            input_dim (int): The number of dimensions of the input space.
            variance (torch.Tensor, optional): Initial value for the variance parameter σ^2.
                Defaults to 1.0 if not provided.
            lengthscale (torch.Tensor, optional): Initial values for the length scale parameters α_q.
                Defaults to 1.0 for each dimension if not provided.
            requires_grad (bool): Whether or not the parameters require gradients. Default is True.
        """
        super(ARDRBFKernel, self).__init__()
        self.input_dim = input_dim

        # Initialize the variance parameter, with gradient computation if required
        if variance is None:
            self._variance = nn.Parameter(torch.ones(1) * 0.8, requires_grad=requires_grad)
        else:
            self._variance = nn.Parameter(variance, requires_grad=requires_grad)

        # Initialize the alpha parameters, one for each input dimension, with gradient computation if required
        if alpha is None:
            self._alpha = nn.Parameter(torch.randn(self.input_dim), requires_grad=requires_grad)
        else:
            self._alpha = nn.Parameter(alpha, requires_grad=requires_grad)

    @property
    def alpha(self):
        # return self._alpha.pow(2).softmax(dim=0)
        return torch.nn.functional.softmax(torch.nn.functional.softplus(self._alpha), dim=0)

    @property
    def variance(self):
        return self._variance.pow(2)

    def forward(self, x1, x2):
        """
        Computes the ARD RBF kernel matrix between two sets of input points.

        Args:
            x1 (torch.Tensor): A tensor of shape (n, d), representing n points in a d-dimensional space.
            x2 (torch.Tensor): A tensor of shape (m, d), representing m points in a d-dimensional space.

        Returns:
            torch.Tensor: The computed ARD RBF kernel matrix of shape (n, m).
        """
        if x1.ndimension() < 3:
            x1 = x1.unsqueeze(0)
        if x2.ndimension() < 3:
            x2 = x2.unsqueeze(0)

        # Ensure batch dimensions can be broadcasted
        if x1.size(0) != x2.size(0):
            if x1.size(0) == 1:
                x1 = x1.expand(x2.size(0), -1, -1)
            elif x2.size(0) == 1:
                x2 = x2.expand(x1.size(0), -1, -1)
            else:
                raise ValueError("Batch dimensions of x1 and x2 are not broadcastable and not singleton")

        # Compute the squared differences between points in x1 and x2, scaled by the inverse of the squared length scales
        scaled_diff = self.alpha * (x1.unsqueeze(2) - x2.unsqueeze(1)) ** 2

        # Compute the kernel matrix using the variance and the scaled differences
        kernel = self.variance * torch.exp(-0.5 * scaled_diff.sum(-1))
        if kernel.size(0) == 1:
            kernel = kernel.squeeze(0)

        return kernel


class ARDRBFKernelOld(nn.Module):

    def __init__(self, input_dim, variance=None, alpha=None, requires_grad=True):

        super(ARDRBFKernelOld, self).__init__()
        self.input_dim = input_dim

        # Initialize the variance parameter, with gradient computation if required
        if variance is None:
            self._variance = nn.Parameter(torch.ones(1), requires_grad=requires_grad)
        else:
            self._variance = nn.Parameter(variance, requires_grad=requires_grad)

        # Initialize the alpha parameters, one for each input dimension, with gradient computation if required
        if alpha is None:
            self._alpha = nn.Parameter(torch.randn(self.input_dim), requires_grad=requires_grad)
        else:
            self._alpha = nn.Parameter(alpha, requires_grad=requires_grad)

    @property
    def alpha(self):
        return self._alpha.pow(2).softmax(dim=0)

    @property
    def variance(self):
        return self._variance.pow(2)

    def forward(self, x1, x2):
        """
        Computes the ARD RBF kernel matrix between two sets of input points.

        Args:
            x1 (torch.Tensor): A tensor of shape (n, d), representing n points in a d-dimensional space.
            x2 (torch.Tensor): A tensor of shape (m, d), representing m points in a d-dimensional space.

        Returns:
            torch.Tensor: The computed ARD RBF kernel matrix of shape (n, m).
        """
        # Compute the squared differences between points in x1 and x2, scaled by the inverse of the squared length scales
        scaled_diff = self.alpha * (x1.unsqueeze(1) - x2.unsqueeze(0)) ** 2

        # Compute the kernel matrix using the variance and the scaled differences
        kernel = self.variance * torch.exp(-0.5 * scaled_diff.sum(-1))
        if kernel.size(0) == 1:
            kernel = kernel.squeeze()

        return kernel


