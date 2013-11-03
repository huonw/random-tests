/// Distributional tests for distributions in the standard lib
use std::num;
use std::rand::{Rng, StdRng};
use std::rand::distributions::{Sample, Exp1, StandardNormal, Gamma};

#[cfg(test)]
use std::rand::distributions::{RandSample};

use kolmogorov_smirnov::ks_unif_test;
use t_test;

pub static SIG: f64 = 0.01;
static NUM_MEANS: uint = 10000;
static EACH_MEAN: uint = 10000;

static KS_SIZE: uint = 10_000_000;

static NUM_MOMENTS: uint = 3;

/// Compute the first NUM_MOMENTS moments of a sample of size `count`
/// from `dist` using `rng` as the source of randomness.
fn moments<S: Sample<f64>, R: Rng>(rng: &mut R,
                                   dist: &mut S, count: uint) -> [f64, .. NUM_MOMENTS] {
    let mut moms = [0., .. NUM_MOMENTS];

    for _ in range(0, count) {
        let v = dist.sample(rng);
        let mut x = v;
        for m in moms.mut_iter() {
            *m = x;
            x *= v;
        }
    }
    moms
}

/// Compute the mean and variance of a sample of size `num_means` of
/// mean and variance estimators of samples of size `each_mean` from
/// `dist`. This is designed to exploit the CLT for the mean and
/// variance estimators of `dist`, so it assumes that the 4th moment
/// of the distribution exists.
///
/// Returns sample (mean, variance) for the mean and variance
/// estimators respectively.
fn mean_var_of_moments<S: Sample<f64>>(dist: &mut S,
                                       each_mean: uint,
                                       num_means: uint) -> [(f64, f64), .. NUM_MOMENTS] {
    let mut rng = StdRng::new();

    let mut mean_vars = [(0., 0.), .. NUM_MOMENTS];

    for _ in range(0, each_mean) {
        let mom = moments(&mut rng, dist, each_mean);

        for (&(ref mut s, ref mut s2), &v) in mean_vars.mut_iter().zip(mom.iter()) {
            *s += v;
            *s2 += v * v;
        }
    }
    // centralise the EX and EX^2 moments for each of the moment
    // estimators of the distribution.
    let n = num_means as f64;
    for &(ref mut s, ref mut s2) in mean_vars.mut_iter() {
        let mu = *s / n;
        let var = (*s2 - *s * mu) / (n - 1.);

        *s = mu;
        *s2 = var;
    }

    mean_vars
}

/// Perform a t-test for the moments of `dist` being equal to the
/// values in `expected`. Fail!s when the difference significant at
/// the `SIG` level. (Note: this performs no corrections for multiple
/// testing)
pub fn t_test_mean_var<S: Sample<f64>>(name: &str,
                                       mut dist: S,
                                       expected: &[f64]) {
    assert!(expected.len() >= NUM_MOMENTS);

    let moments = mean_var_of_moments(&mut dist, EACH_MEAN, NUM_MEANS);
    let mut msgs = ~[];
    for (i, (&(mean, var), &expected)) in moments.iter().zip(expected.iter()).enumerate() {
        let pvalue = t_test::t_test(mean, num::sqrt(var), NUM_MEANS, expected);

        if pvalue < SIG {
            msgs.push(format!("reject E[X^{}] {} = {}, p = {} < {}",
                              i + 1,
                              mean, expected,
                              pvalue, SIG))
        }

    }
    if !msgs.is_empty() {
        fail!("{} failed: {}", name, msgs.connect(", "))
    }
}

/// Perform a Kolmogorov-Smirnov test that samples from `dist` are
/// actually from the distribution with the cumulative distribution
/// function `cdf`.
pub fn ks_test_dist<S: Sample<f64>>(name: &str,
                                    mut dist: S,
                                    cdf: &fn(f64) -> f64) {
    let mut rng = StdRng::new();
    let mut v = range(0, KS_SIZE).map(|_| cdf(dist.sample(&mut rng))).to_owned_vec();
    let pvalue = ks_unif_test(v);

    if pvalue < SIG {
        fail!("{} failed: p = {} < {}", name, pvalue, SIG);
    }
}

struct DirectExpSample;
impl Sample<f64> for DirectExpSample {
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 { *r.gen::<Exp1>() }
}
struct DirectNormSample;
impl Sample<f64> for DirectNormSample {
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 { *r.gen::<StandardNormal>() }
}


#[test]
fn t_test_unif() {
    let mut moments = [0., .. NUM_MOMENTS];
    for (i, m) in moments.mut_iter().enumerate() {
        // for U(0, 1), E[X^k] = 1 / (k + 1).
        *m = 1. / (i as f64 + 2.);
    }

    t_test_mean_var("U(0,1)", RandSample::<f64>,
                    moments);
}

#[test]
fn t_test_exp() {
    let mut moments = [0., .. NUM_MOMENTS];
    let mut prod = 1.;
    for (i, m) in moments.mut_iter().enumerate() {
        // for Exp(1), E[X^k] = k!
        prod *= i as f64 + 1.;
        *m = prod
    }
    t_test_mean_var("Exp(1)", DirectExpSample,
                    moments);
}
#[test]
fn t_test_norm() {
    let mut moments = [0., .. NUM_MOMENTS];
    let mut prod = 1.;
    for (i, m) in moments.mut_iter().enumerate() {
        // for N(0, 1), E[X^odd] = 0, and E[X^k] = k!! (product of odd
        // numbers up to k) (k even).
        let k = i + 1;
        if k % 2 == 0 { // only even moments are non-zero
            prod *= k as f64 - 1.;
            *m = prod;
        }
    }
    t_test_mean_var("N(0, 1)", DirectNormSample,
                    moments);
}

fn test_gamma(shape: f64, scale: f64) {
    let mut moments = [0., .. NUM_MOMENTS];
    let mut current_moment = 1.;
    for (i, m) in moments.mut_iter().enumerate() {
        // E[X^k] = scale^k * shape * (shape + 1) * ... * (shape + (k - 1))
        current_moment *= scale * (shape + i as f64);
        *m = current_moment
    }
    t_test_mean_var(format!("Gamma({}, {})", shape, scale),
                    Gamma::new(shape, scale),
                    moments)
}
// separate to get fine-grained failures/parallelism.
#[test]
fn t_test_gamma_very_small() { test_gamma(0.001, 1.) }
#[test]
fn t_test_gamma_small() { test_gamma(0.4, 2.) }
#[test]
fn t_test_gamma_one() { test_gamma(1., 3.) }
#[test]
fn t_test_gamma_large() { test_gamma(2.5, 4.) }
#[test]
fn t_test_gamma_very_large() { test_gamma(1000., 5.) }

#[test]
fn ks_test_unif() {
    ks_test_dist("U(0, 1)", RandSample::<f64>, ::unif_cdf)
}

#[test]
fn ks_test_exp() {
    ks_test_dist("Exp(1)", DirectExpSample, ::exp_cdf)
}

#[test]
fn ks_test_norm() {
    ks_test_dist("N(0, 1)", DirectNormSample, ::normal_cdf)
}

// Don't have the infrastructure (specifically, the CDF is awkward to
// implement) for Kolmogorov-Smirnov of Gamma.
