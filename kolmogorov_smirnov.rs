use std::num;
use extra::sort;

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
    for k in range(1, 10000) {
        let y = (2 * k - 1) as f64 * Real::pi() / statistic;
        sum += num::exp(-y * y / 8.)
    }
    1.0 - sum * num::sqrt(2.0 * Real::pi()) / statistic
}

/// Test `data` for uniformity, returning a p-value based on the
/// (asymptotic) Kolmogorov distribution, so this will give most
/// accurate results for large samples.
pub fn ks_unif_test(data: &mut [f64]) -> f64 {
    sort::quick_sort3(data);

    let n = data.len() as f64;
    let mut sup = 0.0;
    for (i, &x) in data.iter().enumerate() {
        sup = num::max(sup,
                       num::max(num::abs(i as f64 / n - x),
                                num::abs((i + 1) as f64 / n - x)));
    }
    ks_cdf(num::sqrt(n) * sup)
}
