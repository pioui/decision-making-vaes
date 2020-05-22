import numpy as np
from scipy.linalg import sqrtm
from arviz.stats import psislw
from scipy.stats import norm

from sbvae.dataset import SyntheticGaussianDataset


nus = np.geomspace(1e-2, 1e1, num=40)
DIM_Z = 6
DIM_X = 10
DATASET = SyntheticGaussianDataset(dim_z=DIM_Z, dim_x=DIM_X, n_samples=1000, nu=1)


def model_evaluation_loop(
    trainer, eval_encoder, counts_eval, encoder_eval_name,
):
    # posterior query evaluation: groundtruth
    seq = trainer.test_set.sequential(batch_size=10)
    mean = np.dot(DATASET.mz_cond_x_mean, DATASET.X[seq.indices, :].T)[0, :]
    std = np.sqrt(DATASET.pz_condx_var[0, 0])
    exact_cdf = norm.cdf(0, loc=mean, scale=std)

    is_cdf_nus = seq.prob_eval(
        1000,
        nu=nus,
        encoder_key=encoder_eval_name,
        counts=counts_eval,
        z_encoder=eval_encoder,
    )[2]
    exact_cdfs_nus = np.array([norm.cdf(nu, loc=mean, scale=std) for nu in nus]).T

    log_ratios = (
        trainer.test_set.log_ratios(
            n_samples_mc=5000,
            encoder_key=encoder_eval_name,
            counts=counts_eval,
            z_encoder=eval_encoder,
        )
        .detach()
        .numpy()
    )
    # Input should be n_obs, n_samples
    log_ratios = log_ratios.T
    _, khat_vals = psislw(log_ratios)

    # posterior query evaluation: aproposal distribution
    seq_mean, seq_var, is_cdf, ess = seq.prob_eval(
        1000, encoder_key=encoder_eval_name, counts=counts_eval, z_encoder=eval_encoder,
    )

    return {
        "IWELBO": trainer.test_set.iwelbo(
            5000,
            encoder_key=encoder_eval_name,
            counts=counts_eval,
            z_encoder=eval_encoder,
        ),
        "L1_IS_ERRS": np.abs(is_cdf_nus - exact_cdfs_nus).mean(0),
        "KHAT": khat_vals,
        "exact_lls_test": trainer.test_set.exact_log_likelihood(),
        "exact_lls_train": trainer.train_set.exact_log_likelihood(),
        "model_lls_test": trainer.test_set.model_log_likelihood(),
        "model_lls_train": trainer.train_set.model_log_likelihood(),
        # "plugin_cdf": norm.cdf(0, loc=seq_mean[:, 0], scale=np.sqrt(seq_var[:, 0])),
        "l1_err_ex_is": np.mean(np.abs(exact_cdf - is_cdf)),
        "l2_ess": ess,
        "gt_post_var": DATASET.pz_condx_var,
        # "sigma_sqrt": sqrtm(gt_post_var),
    }