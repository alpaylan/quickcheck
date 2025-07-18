use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::hash::BuildHasherDefault;
use std::path::PathBuf;

use crate::tester::ResultStatus;

use super::{quickcheck, Gen, QuickCheck, TestResult};

#[test]
fn prop_oob() {
    fn prop() -> bool {
        let zero: Vec<bool> = vec![];
        zero[0]
    }
    let result = QuickCheck::new().quicktest(prop as fn() -> bool);
    if let ResultStatus::Finished = result.status {
        panic!(
            "prop_oob should fail with a runtime error \
            but instead it passed {} tests.",
            result.n_tests_passed
        );
    }
}

#[test]
fn prop_reverse_reverse() {
    fn prop(xs: Vec<usize>) -> bool {
        let rev: Vec<_> = xs.clone().into_iter().rev().collect();
        let revrev: Vec<_> = rev.into_iter().rev().collect();
        xs == revrev
    }
    quickcheck(prop as fn(Vec<usize>) -> bool);
}

quickcheck! {
    fn prop_reverse_reverse_macro(xs: Vec<usize>) -> bool {
        let rev: Vec<_> = xs.clone().into_iter().rev().collect();
        let revrev: Vec<_> = rev.into_iter().rev().collect();
        xs == revrev
    }

    #[should_panic]
    fn prop_macro_panic(_x: u32) -> bool {
        assert!(false);
        false
    }
}

#[test]
fn reverse_single() {
    fn prop(xs: Vec<usize>) -> TestResult {
        if xs.len() != 1 {
            TestResult::discard()
        } else {
            TestResult::from_bool(
                xs == xs.clone().into_iter().rev().collect::<Vec<_>>(),
            )
        }
    }
    quickcheck(prop as fn(Vec<usize>) -> TestResult);
}

#[test]
fn reverse_app() {
    fn prop(xs: Vec<usize>, ys: Vec<usize>) -> bool {
        let mut app = xs.clone();
        app.extend(ys.iter().copied());
        let app_rev: Vec<usize> = app.into_iter().rev().collect();

        let rxs: Vec<usize> = xs.into_iter().rev().collect();
        let mut rev_app = ys.into_iter().rev().collect::<Vec<usize>>();
        rev_app.extend(rxs);

        app_rev == rev_app
    }
    quickcheck(prop as fn(Vec<usize>, Vec<usize>) -> bool);
}

#[test]
fn max() {
    fn prop(x: isize, y: isize) -> TestResult {
        if x > y {
            TestResult::discard()
        } else {
            TestResult::from_bool(::std::cmp::max(x, y) == y)
        }
    }
    quickcheck(prop as fn(isize, isize) -> TestResult);
}

#[test]
fn sort() {
    fn prop(mut xs: Vec<isize>) -> bool {
        xs.sort_unstable();
        for i in xs.windows(2) {
            if i[0] > i[1] {
                return false;
            }
        }
        true
    }
    quickcheck(prop as fn(Vec<isize>) -> bool);
}

fn sieve(n: usize) -> Vec<usize> {
    if n <= 1 {
        return vec![];
    }

    let mut marked = vec![false; n + 1];
    marked[0] = true;
    marked[1] = true;
    marked[2] = true;
    for p in 2..n {
        for i in (2 * p..n).filter(|&n| n % p == 0) {
            marked[i] = true;
        }
    }
    marked
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { None } else { Some(i) })
        .collect()
}

fn is_prime(n: usize) -> bool {
    n != 0 && n != 1 && (2..).take_while(|i| i * i <= n).all(|i| n % i != 0)
}

#[test]
#[should_panic]
fn sieve_not_prime() {
    fn prop_all_prime(n: u8) -> bool {
        sieve(n as usize).into_iter().all(is_prime)
    }
    quickcheck(prop_all_prime as fn(u8) -> bool);
}

#[test]
#[should_panic]
fn sieve_not_all_primes() {
    fn prop_prime_iff_in_the_sieve(n: u8) -> bool {
        let n = n as usize;
        sieve(n) == (0..=n).filter(|&i| is_prime(i)).collect::<Vec<_>>()
    }
    quickcheck(prop_prime_iff_in_the_sieve as fn(u8) -> bool);
}

#[test]
fn testable_result() {
    fn result() -> Result<bool, String> {
        Ok(true)
    }
    quickcheck(result as fn() -> Result<bool, String>);
}

#[test]
#[should_panic]
fn testable_result_err() {
    quickcheck(Err::<bool, i32> as fn(i32) -> Result<bool, i32>);
}

#[test]
fn testable_unit() {
    fn do_nothing() {}
    quickcheck(do_nothing as fn());
}

#[test]
fn testable_unit_panic() {
    fn panic() {
        panic!();
    }
    assert!(QuickCheck::new().quicktest(panic as fn()).is_err());
}

#[test]
fn regression_issue_83() {
    fn prop(_: u8) -> bool {
        true
    }
    QuickCheck::new()
        .set_rng(Gen::new(1024))
        .quickcheck(prop as fn(u8) -> bool);
}

#[test]
fn regression_issue_83_signed() {
    fn prop(_: i8) -> bool {
        true
    }
    QuickCheck::new()
        .set_rng(Gen::new(1024))
        .quickcheck(prop as fn(i8) -> bool);
}

// Test that we can show the message after panic
#[test]
#[should_panic(expected = "foo")]
fn panic_msg_1() {
    fn prop() -> bool {
        panic!("foo");
    }
    quickcheck(prop as fn() -> bool);
}

#[test]
#[should_panic(expected = "foo")]
fn panic_msg_2() {
    fn prop() -> bool {
        assert!("foo" == "bar");
        true
    }
    quickcheck(prop as fn() -> bool);
}

#[test]
#[should_panic(expected = "foo")]
fn panic_msg_3() {
    fn prop() -> bool {
        assert_eq!("foo", "bar");
        true
    }
    quickcheck(prop as fn() -> bool);
}

#[test]
#[should_panic]
fn regression_issue_107_hang() {
    fn prop(a: Vec<u8>) -> bool {
        a.contains(&1)
    }
    quickcheck(prop as fn(_) -> bool);
}

#[cfg(not(feature = "etna"))]
#[test]
#[should_panic(
    expected = "(Gave up after 0 successful tests and 70 discarded.)"
)]
fn all_tests_discarded_min_tests_passed_set() {
    fn prop_discarded(_: u8) -> TestResult {
        TestResult::discard()
    }

    QuickCheck::new()
        .max_tests(100)
        .tests(1)
        .min_tests_passed(30)
        .quickcheck(prop_discarded as fn(u8) -> TestResult);
}

#[cfg(feature = "etna")]
#[test]
#[should_panic(
    expected = r#"{"counterexample":null,"discarded":70,"error":null,"execution_time":"0ns","generation_time":"0ns","passed":0,"shrinking_time":"0ns","status":"GaveUp","total_time":"#
)]
fn all_tests_discarded_min_tests_passed_set() {
    fn prop_discarded(_: u8) -> TestResult {
        TestResult::discard()
    }

    QuickCheck::new()
        .max_tests(100)
        .tests(1)
        .min_tests_passed(30)
        .quickcheck(prop_discarded as fn(u8) -> TestResult);
}

#[cfg(not(feature = "etna"))]
#[test]
fn all_tests_discarded_min_tests_passed_missing() {
    fn prop_discarded(_: u8) -> TestResult {
        TestResult::discard()
    }

    QuickCheck::new().quickcheck(prop_discarded as fn(u8) -> TestResult);
}

quickcheck! {
    /// The following is a very simplistic test, which only verifies
    /// that our PathBuf::arbitrary does not panic.  Still, that's
    /// something!  :)
    fn pathbuf(_p: PathBuf) -> bool {
        true
    }

    fn basic_hashset(_set: HashSet<u8>) -> bool {
        true
    }

    fn basic_hashmap(_map: HashMap<u8, u8>) -> bool {
        true
    }

    fn substitute_hashset(
        _set: HashSet<u8, BuildHasherDefault<DefaultHasher>>
    ) -> bool {
        true
    }

    fn substitute_hashmap(
        _map: HashMap<u8, u8, BuildHasherDefault<DefaultHasher>>
    ) -> bool {
        true
    }

    fn cstring(_p: CString) -> bool {
        true
    }
}
