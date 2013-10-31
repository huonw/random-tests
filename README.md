# Random tests

[![Build Status](https://travis-ci.org/huonw/random-tests.png)](https://travis-ci.org/huonw/random-tests)

Probabilistic tests for the distributions in Rust's standard
library. These are mostly significance level tests, and so will fail
occasionally (assuming the distributions are implemented
correctly... if they aren't then it will hopefully fail always).

Run with

    rustc --opt-level=3 --test lib.rs && ./lib

This can take a long time, since it uses a lot of random
numbers. Adjust the constants in `std_dists` for more/fewer numbers,
higher is better (but slower).

# TODO

- full of approximations (e.g. it replaces student-t by normal always)
- larger variety of tests (currently just performs t-tests on a set of
  sample means and sample observations to check these are close to the
  expected values)
- should probably use TestU01 and/or Diehard[er] instead of/as well as
  hand-written pure Rust tests.
