use std::num::Float;
use normal_cdf;

/// Perform a two-sided t-test for `mean == expected` based on a
/// sample of size `n` with sample mean `mean` and sample standard
/// deviation `std_dev`. NB. This uses the normal approximation to t
/// for ease of computation, and so should really only be used for
/// large `n`.
pub fn t_test(mean: f64, std_dev: f64, n: uint, expected: f64) -> f64 {
    let t_stat = (mean - expected) / std_dev * (n as f64).sqrt();
    // two sided
    2. * normal_cdf(-t_stat.abs())
}

macro_rules! assert_approx_eq {
    ($lhs:expr, $rhs:expr) => {
        {
            let (lhs, rhs) = ($lhs, $rhs);
            assert!((lhs - rhs).abs() < 1e-4,"assert_approx_eq failed: {} != {}", lhs, rhs);
        }
    }
}

#[test]
fn test_t_test() {
    // exactly equal means
    assert_eq!(t_test(1., 2., 100, 1.), 1.);

    // test created with rnorm(1000) in R. NB. the value from an
    // actual t distribution is 0.01843758.
    assert_approx_eq!(t_test(-0.02484192, 1.006831, 1000, -0.1),
                      0.01824625);
}

#[test]
fn test_normal_cdf() {
    assert_eq!(normal_cdf(0.), 0.5);
    assert_approx_eq!(normal_cdf(1.), 0.8413447);
    assert_approx_eq!(normal_cdf(1.644854), 0.95);
}
