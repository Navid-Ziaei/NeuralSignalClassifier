from src.model.utils.model_utils import *
from src.model.utils.variational import VariationalDist, VariationalLatentVariable, CholeskeyVariationalDist
from torch import optim
import gpytorch
import torch.nn as nn
import math
from sklearn.metrics import classification_report


class GPLVM_Bayesian_DS(nn.Module):
    def __init__(self, y, kernel_reg, likelihood, latent_dim=2, num_inducing_points=10,
                 num_classes=1,
                 inducing_points=None,
                 use_gpytorch=False):
        super(GPLVM_Bayesian_DS, self).__init__()
        self.use_gpytorch = use_gpytorch
        # Parameters
        self.psi0 = None
        self.psi1 = None
        self.psi2 = None

        self.kernel_reg = kernel_reg
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ZeroMean(ard_num_dims=latent_dim)

        if isinstance(kernel_reg, gpytorch.kernels.Kernel):
            self.use_gpytorch_kernel = True
        else:
            self.use_gpytorch_kernel = False

        self.jitter = 1e-4

        self.n = y.shape[0]
        self.m = num_inducing_points
        self.d = y.shape[1]
        self.q = latent_dim
        self.k = num_classes

        self.initialize_variable = True

        self.batch_shape = torch.Size([self.d])

        torch.manual_seed(42)
        if inducing_points is None:
            self.inducing_inputs = nn.Parameter(torch.randn(self.d, num_inducing_points, self.q))
        else:
            self.inducing_inputs = nn.Parameter(inducing_points)

        x_init = torch.nn.Parameter(torch.randn(self.n, self.q))
        if self.use_gpytorch is True:
            x_prior_mean = torch.zeros(self.n, self.q)  # shape: N x Q
            self.prior_x = gpytorch.priors.NormalPrior(x_prior_mean, torch.ones_like(x_prior_mean))
            self.x = gpytorch.models.gplvm.VariationalLatentVariable(self.n, self.d, self.q, x_init, self.prior_x)
            self.q_u = gpytorch.variational.CholeskyVariationalDistribution(self.m, batch_shape=self.batch_shape)
            self.q_f = gpytorch.variational.VariationalStrategy(model=self,
                                                                inducing_points=self.inducing_inputs,
                                                                variational_distribution=self.q_u,
                                                                learn_inducing_locations=True)
        else:
            self.prior_x = torch.distributions.Normal(torch.zeros(self.n, self.q), torch.ones(self.n, self.q))
            self.x = VariationalLatentVariable(self.n, self.d, self.q, X_init=x_init, prior_x=self.prior_x)
            # print(self.x.q_log_sigma)
            # self.q_u = VariationalDist(num_inducing_points=self.m, batch_shape=self.d)
            self.q_u = CholeskeyVariationalDist(num_inducing_points=self.m, batch_shape=self.d)
            self.log_noise_sigma = nn.Parameter(torch.ones(self.d) * -2)

    @property
    def noise_sigma(self):
        return torch.nn.functional.softplus(self.log_noise_sigma)

    def _expand_inputs(self, x, inducing_points):
        """
        Pre-processing step in __call__ to make x the same batch_shape as the inducing points
        """
        batch_shape = torch.broadcast_shapes(inducing_points.shape[:-2], x.shape[:-2])
        inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
        x = x.expand(*batch_shape, *x.shape[-2:])
        return x, inducing_points

    def sample_latent_variable(self):
        return self.x()

    def calculate_psi_1_vect(self):
        # sum1 = ∑_(q=1)^Q(α_q (μ_(n,q)-z_(m,q) )^2)/(α_q S_nq+1)
        squared_dist = (self.x.q_mu.unsqueeze(1) - self.inducing_inputs.unsqueeze(0)) ** 2
        sum1_denum = self.x.q_sigma * self.kernel_reg.alpha.unsqueeze(0) + 1
        sum1_term1 = squared_dist / sum1_denum.unsqueeze(1)
        sum1 = sum1_term1 @ self.kernel_reg.alpha
        # sum2 = ∑_(q=1)^Q log(α_q S_nq+1)
        sum2 = torch.sum(torch.log(sum1_denum), axis=-1)
        # 2 log〖σ_f〗-1/2 sum1 -1/2 sum2
        psi_1 = 2 * torch.log(self.kernel_reg.variance) - 0.5 * sum1 - 0.5 * sum2.unsqueeze(1)

        return torch.exp(psi_1)

    def calculate_psi_2_vect(self):
        # -1/4 ∑_(q=1)^Q▒〖α_q (μ_mq-z_(m^' q) )^2 〗
        squared_dist_z = (self.inducing_inputs.unsqueeze(1) - self.inducing_inputs.unsqueeze(0)) ** 2
        sum1 = 0.25 * squared_dist_z @ self.kernel_reg.alpha

        zbar = (self.inducing_inputs.unsqueeze(1) + self.inducing_inputs.unsqueeze(0)) / 2
        squared_dist_mu_zbar = (self.x.q_mu.unsqueeze(1).unsqueeze(1) - zbar.unsqueeze(0)) ** 2
        sum2_denum = 2 * self.kernel_reg.alpha.unsqueeze(0) * self.x.q_sigma + 1
        sum2 = (squared_dist_mu_zbar / sum2_denum.unsqueeze(1).unsqueeze(1)) @ self.kernel_reg.alpha

        sum3 = torch.sum(0.5 * torch.log(sum2_denum), axis=-1)
        # 4 log〖σ_f〗
        log_psi_2_n = 4 * torch.log(self.kernel_reg.variance) - sum1.unsqueeze(0) - sum2 - sum3.unsqueeze(1).unsqueeze(
            1)
        psi_2 = torch.sum(torch.exp(log_psi_2_n), axis=0)

        return psi_2

    def compute_statistics(self):
        x_samples = self.sample_latent_variable()

        K_nn = self.kernel_reg(x_samples, x_samples)
        K_nm = self.kernel_reg(x_samples, self.inducing_inputs)
        K_mn = K_nm.permute(0, 2, 1)

        self.psi0 = K_nn.diagonal(dim1=1, dim2=2).mean(dim=0).sum()
        self.psi1 = K_nm.mean(dim=0)
        self.psi2 = (K_mn @ K_nm).mean(dim=0)

    def compute_statistics2(self):

        self.psi0 = self.n * self.kernel_reg.variance
        self.psi1 = self.calculate_psi_1_vect()
        self.psi2 = self.calculate_psi_2_vect()

    def ell_reg(self, y_n, K_mm):
        self.compute_statistics()
        # ELL_{i,d}^{reg}=\log{\mathcal{N}\left(y_{id}^n|E_{q_\phi\left(x_i\right)}\left[K_{nm}\right]K_{mm}^{-1}m_d,\sigma_y^2\right)}
        # -\frac{1}{2\sigma_y^2}Tr\left(E_{q_\phi\left(x_i\right)}\left[K_{nn}\right]\right)+\frac{1}{2\sigma_y^2}TrKmm-1Eqϕxi[KmnKnm]
        # -\frac{1}{2\sigma_y^2}Tr\left(S_dK_{mm}^{-1}E_{q_\phi\left(x_i\right)}\left[K_{mn}K_{nm}\right]K_{mm}^{-1}\right)

        # Compute the inverse of K_mm
        K_mm_inv = torch.inverse(K_mm)

        # Compute the mean of the normal distribution
        mean = self.psi1 @ K_mm_inv @ self.q_u.mu

        # Compute the normal distribution and log likelihood
        normal_dist = torch.distributions.Normal(mean, self.noise_sigma)
        log_likelihood = normal_dist.log_prob(y_n).sum() / (self.d * self.n)

        # Compute the trace terms
        trace_term_1 = (-0.5 / self.noise_sigma.pow(2)) * self.psi0.sum() / self.n
        trace_term_2 = (0.5 / self.noise_sigma.pow(2)) * torch.trace(K_mm_inv * self.psi2) / self.n
        trace_term_3 = 0
        for d in range(self.d):
            trace_term_3 = 0.5 / self.noise_sigma.pow(2) * torch.trace(
                torch.diag(self.q_u.sigma[:, d]) @ K_mm_inv @ self.psi2 @ K_mm_inv) / self.n
        trace_term_3 /= self.d

        # Compute ELL_{i,d}^{reg}
        return log_likelihood + trace_term_1 + trace_term_2 + trace_term_3

    def ell_reg2(self, y_n, k_mm):
        self.compute_statistics2()
        I_n = torch.eye(self.n)
        beta = 1 / self.noise_sigma.pow(2)
        # W=βI_N-β^2 ψ_1 (βψ^2+K_MM )^T ψ_1^T
        cov1 = beta * self.psi2 + k_mm
        w = beta * I_n - self.psi1 @ torch.inverse(cov1) @ self.psi1.t()
        # N/2  log[β]
        F = 0.5 * self.n * torch.log(beta)
        # 1/2  log|K_MM|
        F += 0.5 * torch.logdet(k_mm)
        # -N/2  log 2π
        F -= 0.5 * self.n * torch.log(torch.tensor(np.pi))
        # -1/2  log|βψ_2+K_MM|
        F -= 0.5 * torch.logdet(cov1)

        # -(βψ_0)/2
        F -= 0.5 * beta * self.psi0
        # β/2 Tr(K_MM^(-1) ψ_2 )
        F += 0.5 * torch.trace(torch.inverse(k_mm) @ self.psi2)
        # -1/2 y_d^T Wy_d
        F = F * self.d - 0.5 * torch.trace(y_n.t() @ w @ y_n)

        return F / (self.n * self.d)

    def ell_cls(self):
        kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return 0

    def elbo(self, x, y):
        batch_size = x.shape[0]
        if self.use_gpytorch is True:
            if self.initialize_variable is True:
                self.q_u.initialize_variational_distribution(self.q_f.prior_distribution)
                self.initialize_variable = False

            inducing_points = self.inducing_inputs
            if inducing_points.shape[:-2] != x.shape[:-2]:
                x, inducing_points = self._expand_inputs(x, inducing_points)

            predictive_dist = self.q_f(x=x)

            ell_reg = self.likelihood.expected_log_prob(y.t(), predictive_dist).sum(-1).div(batch_size)  # of shape [D]
            kl_u = self.q_f.kl_divergence().div(self.n)
            kl_x = self.x._added_loss_terms['x_kl'].loss()
        else:
            x, z = self._expand_inputs(x, self.inducing_inputs)

            if self.use_gpytorch_kernel is False:
                k_nn_reg = self.kernel_reg(x, x) + torch.eye(batch_size).unsqueeze(0) * self.jitter  # of size [D, N, N]
                k_mm_reg = self.kernel_reg(self.inducing_inputs, self.inducing_inputs)  # of size [D, M, M]
                k_mn_reg = self.kernel_reg(self.inducing_inputs, x)
            else:
                k_nn_reg = self.kernel_reg(x, x).evaluate() + torch.eye(batch_size).unsqueeze(
                    0) * self.jitter  # of size [D, N, N]
                k_mm_reg = self.kernel_reg(self.inducing_inputs, self.inducing_inputs).evaluate()  # of size [D, M, M]
                k_mn_reg = self.kernel_reg(self.inducing_inputs, x).evaluate()

            k_mm_reg += torch.eye(self.m).unsqueeze(0) * self.jitter

            predictive_dist = self.predictive_distribution(k_nn_reg, k_mm_reg, k_mn_reg)
            ell_reg = self.expected_log_prob_reg(y, predictive_dist).sum(0).div(batch_size)
            prior_p_u_reg = torch.distributions.MultivariateNormal(torch.zeros(self.m, ), k_mm_reg)
            kl_u = self.q_u.kl(prior_u=prior_p_u_reg)
            kl_x = self.x.kl_loss
        loss = ell_reg - kl_u - kl_x
        return loss.sum()

    def expected_log_prob_reg(self, target, predictive_dist):
        mean, variance = predictive_dist.mean.t(), predictive_dist.variance.t()
        num_points, num_dims = mean.shape
        # Potentially reshape the noise to deal with the multitask case
        noise = self.noise_sigma.unsqueeze(0)

        res = ((target - mean).square() + variance) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)

        return res

    def predictive_distribution(self, k_nn, k_mm, k_mn, whitening_parameters=True):
        # k_mm = LL^T
        L = torch.linalg.cholesky(k_mm)  # torch.cholesky(k_mm, upper=False)
        m_d = self.q_u.mu  # of size [D, M, 1]
        if len(self.q_u.sigma.shape) > 2 and self.q_u.sigma.shape[1] == self.q_u.sigma.shape[2]:
            s_d = self.q_u.sigma
        else:
            s_d = torch.diag_embed(self.q_u.sigma)  # of size [D, M, M] It's a eye matrix

        prior_dist_co = torch.eye(self.m)

        if whitening_parameters is True:
            # A = A=L^(-1) K_MN  (interp_term)
            interp_term = torch.linalg.solve(L, k_mn)
            # μ_f=A^T m_d^'
            # Σ_f=A^T (S'-I)A
            predictive_mean = (interp_term.transpose(-1, -2) @ m_d.unsqueeze(-1)).squeeze(-1)  # of size [D, N]
            predictive_covar = interp_term.transpose(-1, -2) @ (s_d - prior_dist_co.unsqueeze(0)) @ interp_term
        else:
            # m_f = K_NM K_MM^(-1) m_d
            # sigma_f = K_NM K_MM^(-1) (S_d-K_MM ) K_MM^(-1) K_MN
            interp_term = torch.cholesky_solve(k_mn, L, upper=False)
            predictive_mean = (interp_term.transpose(-1, -2) @ m_d.unsqueeze(-1)).squeeze(-1)  # of size [D, N]
            predictive_covar = interp_term.transpose(-1, -2) @ (s_d - k_mm) @ interp_term
        predictive_covar += k_nn

        return torch.distributions.MultivariateNormal(predictive_mean, predictive_covar)

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.kernel_reg(X)
        dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)

    def train_model(self, y, learning_rate=0.01, epochs=100, batch_size=100):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        list_alpha = []
        for epoch in range(epochs):
            batch_index = self._get_batch_idx(batch_size)
            optimizer.zero_grad()
            x = self.sample_latent_variable()
            sample_batch = x[batch_index]
            loss = -self.elbo(sample_batch, y[batch_index])
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        plot_loss(losses)

        return losses


class GPLVM_Bayesian(nn.Module):
    def __init__(self, y, kernel, q=2, m=10, inducing_points=None):
        super(GPLVM_Bayesian, self).__init__()

        # Parameters
        self.psi0 = None
        self.psi1 = None
        self.psi2 = None
        self.kernel = kernel
        self.jitter = 1e-6

        self.n = y.shape[0]
        self.m = m
        self.d = y.shape[1]
        self.q = q
        self._noise_var = nn.Parameter(torch.ones(1) * 0.1)

        if inducing_points is None:
            self.z = nn.Parameter(torch.randn(m, self.q))
        else:
            self.z = nn.Parameter(inducing_points)

        self.q_mu = nn.Parameter(torch.zeros(self.n, q))
        self.q_log_sigma = nn.Parameter(torch.ones(self.n, q))

    @property
    def noise_var(self):
        return torch.nn.functional.softplus(self.log_noise_var)

    @property
    def q_sigma(self):
        return torch.nn.functional.softplus(self.q_log_sigma)

    def kl_divergence_gaussian_vect(self):
        """
        Computes the KL divergence between two multivariate Gaussians.

        Args:
        - mu_0, mu_1 (torch.Tensor): Mean vectors of the Gaussians.
        - Sigma_0, Sigma_1 (torch.Tensor): Covariance matrices of the Gaussians.

        Returns:
        - KL divergence (torch.Tensor)
        """

        term1 = torch.sum(torch.log(self.q_sigma))
        term2 = torch.sum(self.q_sigma)
        term3 = torch.sum(self.q_mu.pow(2))

        kl = 0.5 * (-term1 - self.n * self.q + term2 + term3)

        return kl

    def calculate_psi_1(self):
        psi_1 = torch.zeros((self.n, self.m))
        for n in range(self.n):
            for m in range(self.m):
                sum1 = torch.sum(
                    self.kernel.alpha * (self.q_mu[n, :] - self.z[m, :]) ** 2 / (
                            self.kernel.alpha * self.q_sigma[n, :] + 1))
                sum2 = torch.sum(torch.log(self.kernel.alpha * self.q_sigma[n, :] + 1))
                psi_1[n, m] = torch.log(self.kernel.variance) - 0.5 * sum1 - 0.5 * sum2
        return torch.exp(psi_1)

    def calculate_psi_1_vect(self):
        # sum1 = ∑_(q=1)^Q(α_q (μ_(n,q)-z_(m,q) )^2)/(α_q S_nq+1)
        squared_dist = (self.q_mu.unsqueeze(1) - self.z.unsqueeze(0)) ** 2
        sum1_denum = self.q_sigma * self.kernel.alpha.unsqueeze(0) + 1
        sum1_term1 = squared_dist / sum1_denum.unsqueeze(1)
        sum1 = sum1_term1 @ self.kernel.alpha
        # sum2 = ∑_(q=1)^Q log(α_q S_nq+1)
        sum2 = torch.sum(torch.log(sum1_denum), axis=-1)
        # 2 log〖σ_f〗-1/2 sum1 -1/2 sum2
        psi_1 = 2 * torch.log(self.kernel.variance) - 0.5 * sum1 - 0.5 * sum2.unsqueeze(1)

        return torch.exp(psi_1)

    def calculate_psi_2(self):
        psi_2_n = torch.zeros((self.n, self.m, self.m))
        for n in range(self.n):
            for m in range(self.m):
                for m_prime in range(self.m):
                    sum1 = (1 / 4) * torch.sum((self.q_mu[m, :] - self.z[m_prime, :]) ** 2 @ self.kernel.alpha)
                    sum2 = torch.sum(
                        (self.kernel.alpha * (self.q_mu[n, :] - (self.z[m, :] + self.z[m_prime, :]) / 2) ** 2) / (
                                2 * self.kernel.alpha * self.q_sigma[n, :] + 1))
                    sum3 = (1 / 2) * torch.sum(torch.log(2 * self.kernel.alpha * self.q_sigma[n, :] + 1))
                    psi_2_n[n, m, m_prime] = 4 * torch.log(self.kernel.variance) - sum1 - sum2 - sum3

        # Calculate ψ_2
        psi_2 = torch.sum(torch.exp(psi_2_n), axis=0)
        return psi_2

    def calculate_psi_2_vact(self):
        # -1/4 ∑_(q=1)^Q▒〖α_q (μ_mq-z_(m^' q) )^2 〗
        squared_dist_z = (self.z.unsqueeze(1) - self.z.unsqueeze(0)) ** 2
        sum1 = 0.25 * squared_dist_z @ self.kernel.alpha

        zbar = (self.z.unsqueeze(1) + self.z.unsqueeze(0)) / 2
        squared_dist_mu_zbar = (self.q_mu.unsqueeze(1).unsqueeze(1) - zbar.unsqueeze(0)) ** 2
        sum2_denum = 2 * self.kernel.alpha.unsqueeze(0) * self.q_sigma + 1
        sum2 = (squared_dist_mu_zbar / sum2_denum.unsqueeze(1).unsqueeze(1)) @ self.kernel.alpha

        sum3 = torch.sum(0.5 * torch.log(sum2_denum), axis=-1)
        # 4 log〖σ_f〗
        log_psi_2_n = 4 * torch.log(self.kernel.variance) - sum1.unsqueeze(0) - sum2 - sum3.unsqueeze(1).unsqueeze(1)
        psi_2 = torch.sum(torch.exp(log_psi_2_n), axis=0)

        return psi_2

    def elbo(self, y):
        log_likelihoods = []
        k_mm = self.kernel(self.z, self.z)

        # KL(P(X)||q(X))
        q_x = torch.distributions.Normal(self.q_mu, self.q_sigma)
        p_x = torch.distributions.Normal(torch.zeros_like(self.q_mu), torch.ones_like(self.q_sigma))
        kl_per_latent_dim = torch.distributions.kl_divergence(q_x, p_x).sum(axis=0)  # vector of size latent_dim
        kl_per_point = kl_per_latent_dim.sum() / self.n  # scalar
        kl_term = kl_per_point / self.d

        # kl_term = self.kl_divergence_gaussian_vect()

        self.psi0 = self.n * self.kernel.variance
        self.psi1 = self.calculate_psi_1_vect()
        self.psi2 = self.calculate_psi_2_vact()

        """for d in range(self.d):
            y_d = y

            # Multivariate normal with zero mean

            log_likelihoods.append(self.calculate_F(y_d, k_mm=k_mm))"""

        return self.calculate_F(y, k_mm=k_mm) - kl_term

    def calculate_F(self, y_d, k_mm):
        I_n = torch.eye(self.n)
        beta = 1 / self.noise_var
        # W=βI_N-β^2 ψ_1 (βψ^2+K_MM )^T ψ_1^T
        cov1 = beta * self.psi2 + k_mm
        w = beta * I_n - self.psi1 @ torch.inverse(cov1) @ self.psi1.t()
        # N/2  log[β]
        F = 0.5 * self.n * torch.log(beta)
        # 1/2  log|K_MM|
        F += 0.5 * torch.logdet(k_mm)
        # -N/2  log 2π
        F -= 0.5 * self.n * torch.log(torch.tensor(np.pi))
        # -1/2  log|βψ_2+K_MM|
        F -= 0.5 * torch.logdet(cov1)

        # -(βψ_0)/2
        F -= 0.5 * beta * self.psi0
        # β/2 Tr(K_MM^(-1) ψ_2 )
        F += 0.5 * torch.trace(torch.inverse(k_mm) @ self.psi2)
        # -1/2 y_d^T Wy_d
        F = F * self.d - 0.5 * torch.trace(y_d.t() @ w @ y_d)

        return F / (self.n * self.d)

    def forward(self, x_star):
        # Since this is a latent model, the forward pass would require the prediction mechanism for GPLVM.
        # This is just a simple example to compute the mean for given inputs x_star
        K_xx = self.kernel(self.x, self.x) + self.jitter * torch.eye(self.n)
        K_xs = self.kernel(self.x, x_star)
        K_ss = self.kernel(x_star, x_star)

        K_xx_inv = torch.inverse(K_xx)

        mean = torch.mm(torch.mm(K_xs.t(), K_xx_inv), self.y)

        return mean

    def train_model(self, y, learning_rate=0.01, epochs=100):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = -self.elbo(y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        plot_loss(losses)

        return losses
