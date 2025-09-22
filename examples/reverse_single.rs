#![allow(dead_code)]
#![allow(unused_imports)]

use quickcheck::{quickcheck, TestResult};

fn reverse<T: Clone>(xs: &[T]) -> Vec<T> {
    let mut rev = vec![];
    for x in xs {
        rev.insert(0, x.clone());
    }
    rev
}

fn main() {
    fn prop(xs: Vec<isize>) -> TestResult {
        if xs.len() != 1 {
            return TestResult::discard();
        }
        TestResult::from_bool(xs == reverse(&xs))
    }

    #[cfg(not(feature = "etna"))]
    quickcheck(prop as fn(Vec<isize>) -> TestResult);
}
