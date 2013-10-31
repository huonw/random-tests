/// Distributional tests for distributions in the standard lib
use std::num;
use std::rand::{Rng, StdRng};
use std::rand::distributions::Sample;

#[cfg(test)]
use std::rand::distributions;
#[cfg(test)]
use std::rand::distributions::RandSample;

use t_test;

pub static SIG: f64 = 0.01;
static NUM_MEANS: uint = 10000;
static EACH_MEAN: uint = 10000;

/// Convert ∑ X_i, ∑ X_i^2 of a sample of size `n` into sample mean and
/// variance.
fn centralise_sample_moments(s: f64, s2: f64, n: uint) -> (f64, f64) {
    let n = n as f64;
    let mu = s / n;
    (mu, (s2 - s * mu) / (n - 1.))
}

/// Compute the sample mean and variance of a sample of size `count`
/// from `dist` using `rng` as the source of randomness.
fn mean_var<S: Sample<f64>, R: Rng>(rng: &mut R, dist: &mut S, count: uint) -> (f64, f64) {
    let mut s = 0.;
    let mut s2 = 0.;

    for _ in range(0, count) {
        let v = dist.sample(rng);
        s += v;
        s2 += v * v;
    }
    centralise_sample_moments(s, s2, count)
}

/// Compute the mean and variance of a sample of size `num_means` of
/// mean and variance estimators of samples of size `each_mean` from
/// `dist`. This is designed to exploit the CLT for the mean and
/// variance estimators of `dist`, so it assumes that the 4th moment
/// of the distribution exists.
///
/// Returns sample (mean, variance) for the mean and variance
/// estimators respectively.
fn mean_var_of_mean_var<S: Sample<f64>>(dist: &mut S,
                                        each_mean: uint, num_means: uint) -> ((f64, f64),
                                                                              (f64, f64)) {
    let mut rng = StdRng::new();

    let mut s_m = 0.;
    let mut s_v = 0.;
    let mut s2_m = 0.;
    let mut s2_v = 0.;
    for _ in range(0, each_mean) {
        let (m, v) = mean_var(&mut rng, dist, each_mean);

        s_m += m;
        s_v += v;

        s2_m += m * m;
        s2_v += v * v;
    }
    (centralise_sample_moments(s_m, s2_m, num_means),
     centralise_sample_moments(s_v, s2_v, num_means))
}

/// Perform a t-test for the mean and variance of `dist` being equal
/// to `expected_mean` and `expected_var` respectively. Fail!s when
/// significant at the `SIG` level.
pub fn t_test_mean_var<S: Sample<f64>>(name: &str,
                                       mut dist: S,
                                       expected_mean: f64, expected_var: f64) {
    let ((m_m, m_v),
         (v_m, v_v)) = mean_var_of_mean_var(&mut dist, EACH_MEAN, NUM_MEANS);

    let m_pvalue = t_test::t_test(m_m, num::sqrt(m_v), NUM_MEANS, expected_mean);
    let v_pvalue = t_test::t_test(v_m, num::sqrt(v_v), NUM_MEANS, expected_var);

    let mut msgs = ~[];
    if m_pvalue < SIG {
        msgs.push(format!("reject mean {} = {}, p = {} < {}",
                          m_m, expected_mean,
                          m_pvalue, SIG))
    }

    if v_pvalue < SIG {
        msgs.push(format!("reject variance {} = {}, p = {} < {}",
                          v_m, expected_var,
                          v_pvalue, SIG))
    }
    if !msgs.is_empty() {
        fail!("{} failed: {}", name, msgs.connect(", "))
    }
}


#[test]
fn t_test_unif() {
    t_test_mean_var("U(0,1)", RandSample::<f64>,
                    0.5, 1. / 12.);
}

#[test]
fn t_test_exp() {
    struct DirectExpSample;
    impl Sample<f64> for DirectExpSample {
        fn sample<R: Rng>(&mut self, r: &mut R) -> f64 { *r.gen::<distributions::Exp1>() }
    }

    t_test_mean_var("Exp(1)", DirectExpSample,
                    1., 1.);
}
#[test]
fn t_test_norm() {
    struct DirectNormSample;
    impl Sample<f64> for DirectNormSample {
        fn sample<R: Rng>(&mut self, r: &mut R) -> f64 { *r.gen::<distributions::StandardNormal>() }
    }

    t_test_mean_var("N(0, 1)", DirectNormSample,
                    0., 1.);
}
