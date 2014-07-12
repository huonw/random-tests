use std::{num};

/// An approximation to the cumulative distribution function of the
/// Kolmogorov distribution. Note that this approximates in a
/// easy/simple/dumb way (truncating the infinite sum at 10000) so
/// doesn't guarantee accuracy, but does guarantee a lack of speed.
pub fn ks_cdf(statistic: f64) -> f64 {
    if statistic <= 0. {
        return 0.;
    }

    let mut sum = 0.0;
    // the longer we go the more accurate we are.
    for k in range(0u, 10000) {
        let y = (2 * k - 1) as f64 * Float::pi() / statistic;
        sum += (-y * y / 8.).exp()
    }
    1.0 - sum * (2.0f64 * Float::pi()).sqrt() / statistic
}

/// Test `data` for uniformity, returning a p-value based on the
/// (asymptotic) Kolmogorov distribution, so this will give most
/// accurate results for large samples.
pub fn ks_unif_test(data: &mut [f64]) -> f64 {
    data.sort_by(|x, y| {
            // arbitrarily decide that NaNs are larger than everything.
            if y.is_nan() {
                Less
            } else if x.is_nan() {
                Greater
            } else if x < y {
                Less
            } else if x == y {
                Equal
            } else {
                Greater
            }
        });

    let n = data.len() as f64;
    let mut sup = 0.0f64;
    for (i, &x) in data.iter().enumerate() {
        sup = sup.max(num::abs(i as f64 / n - x).max(num::abs((i + 1) as f64 / n - x)));
    }
    ks_cdf(n.sqrt() * sup)
}
