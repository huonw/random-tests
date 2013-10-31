#[crate_type = "lib"];

extern mod extra;
use std::num;

pub mod kolmogorov_smirnov;
pub mod t_test;

pub mod std_dists;

/// The CDF of the U(0, 1) distribution.
pub fn unif_cdf(x: f64) -> f64 {
    if x < 0. {
        0.
    } else if x > 1. {
        1.
    } else {
        x
    }
}

/// The CDF of the Exp(1) distribution.
pub fn exp_cdf(x: f64) -> f64 {
    if x < 0. {
        0.
    } else {
        1. - num::exp(-x)
    }
}

/// The CDF of the N(0, 1) distribution.
#[fixed_stack_segment]
pub fn normal_cdf(x: f64) -> f64 {
    return 0.5 * (1.0 + unsafe { erf(x / Real::sqrt2()) });

    #[link_name = "m"]
    extern {
        fn erf(n: f64) -> f64;
    }
}
