import torch
import torch.nn as nn
from linear_operator.operators import CholLinearOperator, TriangularLinearOperator


class VariationalLatentVariable(nn.Module):
    def __init__(self, n, data_dim, latent_dim, X_init, prior_x):
        super(VariationalLatentVariable, self).__init__()
        self.data_dim = data_dim
        self.prior_x = prior_x
        self.n = n
        self.data_dim = data_dim
        # G: there might be some issues here if someone calls .cuda() on their BayesianGPLVM
        # after initializing on the CPU

        # Local variational params per latent point with dimensionality latent_dim
        self.q_mu = torch.nn.Parameter(X_init)
        self.q_log_sigma = torch.nn.Parameter(torch.randn(n, latent_dim))
        self.kl_loss = None
        self.kl_loss_list = []

    @property
    def q_sigma(self):
        return torch.nn.functional.softplus(self.q_log_sigma)

    def forward(self, num_samples=5):
        # Variational distribution over the latent variable q(x)
        q_x = torch.distributions.Normal(self.q_mu, self.q_sigma)

        kl_per_latent_dim = torch.distributions.kl_divergence(q_x, self.prior_x).sum(axis=0)
        # vector of size latent_dim
        kl_per_point = kl_per_latent_dim.sum() / self.n  # scalar
        self.kl_loss = kl_per_point / self.data_dim
        self.kl_loss_list.append(self.kl_loss.detach().numpy())
        return q_x.rsample()

    def kl(self):
        n, q = self.q_mu.shape
        q_x = torch.distributions.Normal(self.q_mu, self.q_sigma)
        kl_per_latent_dim = torch.distributions.kl_divergence(q_x, self.prior_x).sum(
            axis=0)  # vector of size latent_dim
        kl_per_point = kl_per_latent_dim.sum() / n  # scalar
        kl_term = kl_per_point / q
        return kl_term


class VariationalDist(nn.Module):
    def __init__(self, num_inducing_points, batch_shape):
        super(VariationalDist, self).__init__()

        mean_init = torch.zeros(batch_shape, num_inducing_points)
        covar_init = torch.ones(batch_shape, num_inducing_points)

        self.mu = nn.Parameter(mean_init)
        self.log_sigma = nn.Parameter(covar_init)

    @property
    def sigma(self):
        return torch.nn.functional.softplus(self.log_sigma)

    def forward(self):
        # Variational distribution over the latent variable q(x)
        q_x = torch.distributions.Normal(self.mu, self.sigma)
        return q_x.rsample()

    def kl(self, prior_u):
        data_dim, n = self.mu.shape
        q_u = torch.distributions.MultivariateNormal(self.mu, torch.diag_embed(self.sigma))
        kl_per_point = torch.distributions.kl_divergence(q_u, prior_u) / n
        return kl_per_point


class CholeskeyVariationalDist(nn.Module):
    def __init__(self, num_inducing_points, batch_shape):
        super(CholeskeyVariationalDist, self).__init__()

        mean_init = torch.zeros(num_inducing_points)
        covar_init = torch.eye(num_inducing_points, num_inducing_points)

        mean_init = mean_init.repeat(batch_shape, 1)
        covar_init = covar_init.repeat(batch_shape, 1, 1)

        self.register_parameter(name="mu", param=torch.nn.Parameter(mean_init))
        self.register_parameter(name="chol_variational_covar", param=torch.nn.Parameter(covar_init))

    @property
    def sigma(self):
        chol_variational_covar = self.chol_variational_covar

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:]).tril(0)
        chol_variational_covar = TriangularLinearOperator(chol_variational_covar.mul(lower_mask))

        # Now construct the actual matrix
        variational_covar = CholLinearOperator(chol_variational_covar)
        return variational_covar.evaluate()

    @property
    def covar(self):
        chol_variational_covar = self.chol_variational_covar

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:]).tril(0)
        chol_variational_covar = TriangularLinearOperator(chol_variational_covar.mul(lower_mask))

        # Now construct the actual matrix
        variational_covar = CholLinearOperator(chol_variational_covar)
        return variational_covar

    def forward(self):
        chol_variational_covar = self.chol_variational_covar
        dtype = chol_variational_covar.dtype
        device = chol_variational_covar.device

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar = TriangularLinearOperator(chol_variational_covar.mul(lower_mask))

        # Now construct the actual matrix
        variational_covar = CholLinearOperator(chol_variational_covar)
        q_x = torch.distributions.MultivariateNormal(self.mu, variational_covar)
        return q_x.rsample()

    def kl(self, prior_u):
        data_dim, n = self.mu.shape
        chol_variational_covar = self.chol_variational_covar
        dtype = chol_variational_covar.dtype
        device = chol_variational_covar.device

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar = TriangularLinearOperator(chol_variational_covar.mul(lower_mask))

        # Now construct the actual matrix
        variational_covar = CholLinearOperator(chol_variational_covar)
        q_u = torch.distributions.MultivariateNormal(self.mu, variational_covar.evaluate())
        kl_per_point = torch.distributions.kl_divergence(q_u, prior_u) / n
        return kl_per_point
