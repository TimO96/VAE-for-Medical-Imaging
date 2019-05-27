from torch.distributions import Normal, Laplace, Independent, Bernoulli, Gamma, Uniform, Beta

# Distribution namespaces

def normal_dist(mu, var):
    return Normal(loc=mu, scale=var)

def laplace_dist(mu, var):
    return Laplace(loc=mu, scale=var)

def gamma_dist(mu, var):
    return Gamma(concentration=mu, rate=var)

def beta_dist(mu, var):
    return Beta(concentration1=mu, concentration0=var)

def bernoulli_loss(x_hat):
    return Bernoulli(x_hat)

def laplace_loss(x_hat, scale=0.01):
    return Laplace(loc=x_hat, scale=scale)
