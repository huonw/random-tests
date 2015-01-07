#![crate_name="random_test"]
#![crate_type = "lib"]
#![feature(macro_rules, phase)]

#[phase(plugin, link)] extern crate log;

use std::num::Float;

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
        1. - (-x).exp()
    }
}

/// The CDF of the N(0, 1) distribution.
pub fn normal_cdf(x: f64) -> f64 {
    use std::num::Float;
    return 0.5 * (1.0 + unsafe { erf(x / 2.0f64.sqrt()) });

    #[link_name = "m"]
    extern {
        fn erf(n: f64) -> f64;
    }
}
