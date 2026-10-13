import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    """
    KL divergence between:
        P = N(mu_p, sigma_p^2)
        Q = N(mu_q, sigma_q^2)

    D_KL(P || Q)
        = log(sigma_q / sigma_p)
          + [sigma_p^2 + (mu_p - mu_q)^2] / (2 * sigma_q^2)
          - 1/2
    """

    kl = (
        np.log(sigma_q / sigma_p)
        + (sigma_p**2 + (mu_p - mu_q)**2)
        / (2 * sigma_q**2)
        - 0.5
    )

    return float(kl)