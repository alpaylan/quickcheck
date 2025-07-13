use std::cmp;
use std::env;
use std::fmt::Debug;
use std::panic;
use std::time::Duration;

#[cfg(feature = "etna")]
use erased_serde::Serialize;
#[cfg(feature = "etna")]
use serde_sexpr;

use crate::{
    tester::Status::{Discard, Fail, Pass},
    Arbitrary, Gen,
};

/// The main `QuickCheck` type for setting configuration and running
/// `QuickCheck`.
pub struct QuickCheck {
    tests: u64,
    max_tests: u64,
    min_tests_passed: u64,
    rng: Gen,
    max_time: Duration,
}

fn qc_tests() -> u64 {
    let default = 100;
    match env::var("QUICKCHECK_TESTS") {
        Ok(val) => val.parse().unwrap_or(default),
        Err(_) => default,
    }
}

fn qc_max_tests() -> u64 {
    let default = 10_000;
    match env::var("QUICKCHECK_MAX_TESTS") {
        Ok(val) => val.parse().unwrap_or(default),
        Err(_) => default,
    }
}

fn qc_gen_size() -> usize {
    let default = 100;
    match env::var("QUICKCHECK_GENERATOR_SIZE") {
        Ok(val) => val.parse().unwrap_or(default),
        Err(_) => default,
    }
}

fn qc_min_tests_passed() -> u64 {
    let default = 0;
    match env::var("QUICKCHECK_MIN_TESTS_PASSED") {
        Ok(val) => val.parse().unwrap_or(default),
        Err(_) => default,
    }
}

fn qc_max_time() -> std::time::Duration {
    let default = std::time::Duration::from_secs(60);
    match env::var("QUICKCHECK_MAX_TIME") {
        Ok(val) => {
            let secs: u64 = val.parse().unwrap_or(default.as_secs());
            std::time::Duration::from_secs(secs)
        }
        Err(_) => default,
    }
}

impl Default for QuickCheck {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResultStatus {
    /// Exceeds the maximum number of passed tests.
    Finished,
    /// Exceeded maximum number of discards.
    GaveUp,
    /// Exceeded maximum time limit.
    TimedOut,
    /// The test failed with a counterexample.
    Failed { arguments: Vec<String>, err: Option<String> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuickCheckResult {
    /// The number of tests that passed.
    pub n_tests_passed: u64,
    /// The number of tests that were discarded.
    pub n_tests_discarded: u64,
    /// Counterexample
    pub status: ResultStatus,
    /// The time taken to run the tests.
    pub total_time: Duration,
    /// Total time taken to generate the tests.
    pub generation_time: Duration,
    /// Total time taken to execute the tests.
    pub execution_time: Duration,
    /// Total time taken to shrink the tests.
    pub shrinking_time: Duration,
}

impl QuickCheckResult {
    pub fn is_err(&self) -> bool {
        matches!(self.status, ResultStatus::Failed { .. })
    }

    pub fn unwrap_err(self) -> TestResult {
        match self.status {
            ResultStatus::Failed { arguments, err } => TestResult {
                status: Fail,
                arguments: Some(arguments),
                err,
                generation_time: self.generation_time,
                execution_time: self.execution_time,
                shrinking_time: self.shrinking_time,
            },
            _ => panic!("QuickCheckResult does not contain an error"),
        }
    }
}

impl QuickCheckResult {
    /// Returns the human-readable status of the test result.
    #[cfg(not(feature = "etna"))]
    pub fn print_status(&self) {
        match self.status {
            ResultStatus::Finished => {
                info!("(Passed {} QuickCheck tests.)", self.n_tests_passed);
            }
            ResultStatus::GaveUp => {
                panic!(
                    "(Gave up after {} successful tests and {} discarded.)",
                    self.n_tests_passed, self.n_tests_discarded
                );
            }
            ResultStatus::TimedOut => {
                panic!("(Timed out at {} seconds after {} successful tests and {} discarded.)", self.total_time.as_secs(), self.n_tests_passed, self.n_tests_discarded);
            }
            ResultStatus::Failed { ref arguments, ref err } => {
                let mut tr = TestResult::from_bool(false);
                tr.arguments = Some(arguments.clone());
                tr.err = err.clone();
                panic!("{}", tr.failed_msg());
            }
        }
    }
    #[cfg(feature = "etna")]
    // Prints the status of the test result in JSON format.
    pub fn print_status(&self) {
        let result = serde_json::json!({
            "status": match self.status {
                ResultStatus::Finished => "Finished",
                ResultStatus::GaveUp => "GaveUp",
                ResultStatus::TimedOut => "TimedOut",
                ResultStatus::Failed { .. } => "Failed",
            },
            "tests": self.n_tests_passed,
            "discards": self.n_tests_discarded,
            "time": format!("{}ns", self.total_time.as_nanos()),
            "generation_time": format!("{}ns", self.generation_time.as_nanos()),
            "execution_time": format!("{}ns", self.execution_time.as_nanos()),
            "shrinking_time": format!("{}ns", self.shrinking_time.as_nanos()),
            "counterexample": match self.status {
                ResultStatus::Failed { ref arguments, err: _ } => {
                    Some(format!("({})", arguments.join(" ")))
                }
                _ => None
            },
            "error": match self.status {
                ResultStatus::Failed { err: Some(ref e), .. } => Some(e.clone()),
                _ => None
            }
        });

        let message = serde_json::to_string(&result)
            .unwrap_or_else(|_| "Failed to serialize result".to_string());

        println!("[|{message}|]");
    }
}

impl QuickCheck {
    /// Creates a new `QuickCheck` value.
    ///
    /// This can be used to run `QuickCheck` on things that implement
    /// `Testable`. You may also adjust the configuration, such as the
    /// number of tests to run.
    ///
    /// By default, the maximum number of passed tests is set to `100`, the max
    /// number of overall tests is set to `10000` and the generator is created
    /// with a size of `100`.
    pub fn new() -> QuickCheck {
        let rng = Gen::new(qc_gen_size());
        let tests = qc_tests();
        let max_tests = cmp::max(tests, qc_max_tests());
        let min_tests_passed = qc_min_tests_passed();
        let max_time = qc_max_time();

        QuickCheck { tests, max_tests, min_tests_passed, rng, max_time }
    }

    /// Set the random number generator to be used by `QuickCheck`.
    pub fn set_rng(self, rng: Gen) -> QuickCheck {
        QuickCheck { rng, ..self }
    }

    #[deprecated(since = "1.1.0", note = "use `set_rng` instead")]
    pub fn r#gen(self, rng: Gen) -> QuickCheck {
        self.set_rng(rng)
    }

    /// Set the number of tests to run.
    ///
    /// This actually refers to the maximum number of *passed* tests that
    /// can occur. Namely, if a test causes a failure, future testing on that
    /// property stops. Additionally, if tests are discarded, there may be
    /// fewer than `tests` passed.
    pub fn tests(mut self, tests: u64) -> QuickCheck {
        self.tests = tests;
        self
    }

    /// Set the maximum number of tests to run.
    ///
    /// The number of invocations of a property will never exceed this number.
    /// This is necessary to cap the number of tests because `QuickCheck`
    /// properties can discard tests.
    pub fn max_tests(mut self, max_tests: u64) -> QuickCheck {
        self.max_tests = max_tests;
        self
    }

    /// Set the maximum time to run tests.
    ///
    /// This is useful for long-running tests that may take a while to
    /// complete. If the time limit is reached, the tests will stop
    /// running, and the results will be returned up to that point.
    #[cfg(feature = "etna")]
    pub fn max_time(mut self, max_time: std::time::Duration) -> QuickCheck {
        self.max_time = max_time;
        self
    }

    /// Set the minimum number of tests that needs to pass.
    ///
    /// This actually refers to the minimum number of *valid* *passed* tests
    /// that needs to pass for the property to be considered successful.
    pub fn min_tests_passed(mut self, min_tests_passed: u64) -> QuickCheck {
        self.min_tests_passed = min_tests_passed;
        self
    }

    /// Tests a property and returns the result.
    ///
    /// The result returned is either the number of tests passed or a witness
    /// of failure.
    ///
    /// (If you're using Rust's unit testing infrastructure, then you'll
    /// want to use the `quickcheck` method, which will `panic!` on failure.)
    pub fn quicktest<A>(&mut self, f: A) -> QuickCheckResult
    where
        A: Testable,
    {
        let mut n_tests_passed = 0;
        let mut n_tests_discarded = 0;
        let mut total_generation_time = std::time::Duration::default();
        let mut total_execution_time = std::time::Duration::default();
        let mut total_shrinking_time = std::time::Duration::default();

        let start = std::time::Instant::now();
        for _ in 0..self.max_tests {
            let result = f.result(&mut self.rng);
            total_generation_time += result.generation_time;
            total_execution_time += result.execution_time;
            total_shrinking_time += result.shrinking_time;

            match result {
                TestResult { status: Pass, .. } => n_tests_passed += 1,
                TestResult { status: Discard, .. } => n_tests_discarded += 1,
                r @ TestResult { status: Fail, .. } => {
                    return QuickCheckResult {
                        n_tests_passed,
                        n_tests_discarded,
                        status: ResultStatus::Failed {
                            arguments: r.arguments.unwrap_or_default(),
                            err: r.err,
                        },
                        total_time: start.elapsed(),
                        generation_time: total_generation_time,
                        execution_time: total_execution_time,
                        shrinking_time: total_shrinking_time,
                    }
                }
            }

            if start.elapsed() >= self.max_time {
                return QuickCheckResult {
                    n_tests_passed,
                    n_tests_discarded,
                    status: ResultStatus::TimedOut,
                    total_time: start.elapsed(),
                    generation_time: total_generation_time,
                    execution_time: total_execution_time,
                    shrinking_time: total_shrinking_time,
                };
            }

            if n_tests_passed >= self.tests {
                break;
            }

            if n_tests_discarded >= self.max_tests - self.min_tests_passed {
                // If min_tests_passed is set to 0, discards do not panic, so we fallback to
                // the default finished status.
                if self.min_tests_passed == 0 {
                    break;
                }

                return QuickCheckResult {
                    n_tests_passed,
                    n_tests_discarded,
                    status: ResultStatus::GaveUp,
                    total_time: start.elapsed(),
                    generation_time: total_generation_time,
                    execution_time: total_execution_time,
                    shrinking_time: total_shrinking_time,
                };
            }
        }

        QuickCheckResult {
            n_tests_passed,
            n_tests_discarded,
            status: ResultStatus::Finished,
            total_time: start.elapsed(),
            generation_time: total_generation_time,
            execution_time: total_execution_time,
            shrinking_time: total_shrinking_time,
        }
    }

    #[cfg(feature = "etna")]
    pub fn quicksample<A>(
        &mut self,
        f: A,
    ) -> Vec<(std::time::Duration, String)>
    where
        A: Sample,
    {
        let t0 = std::time::Instant::now();
        let mut results = vec![];
        let mut n_tests_passed = 0;
        while n_tests_passed < self.tests && t0.elapsed() < self.max_time {
            let (generation_time, args) = f.sample(&mut self.rng);
            results.push((generation_time, format!("({})", args.join(" "))));
            n_tests_passed += 1;
        }

        results
    }

    /// Tests a property and calls `panic!` on failure.
    ///
    /// The `panic!` message will include a (hopefully) minimal witness of
    /// failure.
    ///
    /// It is appropriate to use this method with Rust's unit testing
    /// infrastructure.
    ///
    /// Note that if the environment variable `RUST_LOG` is set to enable
    /// `info` level log messages for the `quickcheck` crate, then this will
    /// include output on how many `QuickCheck` tests were passed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use quickcheck::QuickCheck;
    ///
    /// fn prop_reverse_reverse() {
    ///     fn revrev(xs: Vec<usize>) -> bool {
    ///         let rev: Vec<_> = xs.clone().into_iter().rev().collect();
    ///         let revrev: Vec<_> = rev.into_iter().rev().collect();
    ///         xs == revrev
    ///     }
    ///     QuickCheck::new().quickcheck(revrev as fn(Vec<usize>) -> bool);
    /// }
    /// ```
    pub fn quickcheck<A>(&mut self, f: A)
    where
        A: Testable,
    {
        // Ignore log init failures, implying it has already been done.
        let _ = crate::env_logger_init();

        let result = self.quicktest(f);

        result.print_status();
    }
}

/// Convenience function for running `QuickCheck`.
///
/// This is an alias for `QuickCheck::new().quickcheck(f)`.
pub fn quickcheck<A: Testable>(f: A) {
    QuickCheck::new().quickcheck(f)
}

/// Describes the status of a single instance of a test.
///
/// All testable things must be capable of producing a `TestResult`.
#[derive(Clone, Debug, PartialEq)]
pub struct TestResult {
    status: Status,
    arguments: Option<Vec<String>>,
    err: Option<String>,
    generation_time: std::time::Duration,
    execution_time: std::time::Duration,
    shrinking_time: std::time::Duration,
}

/// Whether a test has passed, failed or been discarded.
#[derive(Clone, Debug, PartialEq)]
enum Status {
    Pass,
    Fail,
    Discard,
}

impl TestResult {
    /// Produces a test result that indicates the current test has passed.
    pub fn passed() -> TestResult {
        TestResult::from_bool(true)
    }

    /// Produces a test result that indicates the current test has failed.
    pub fn failed() -> TestResult {
        TestResult::from_bool(false)
    }

    /// Produces a test result that indicates failure from a runtime error.
    pub fn error<S: Into<String>>(msg: S) -> TestResult {
        let mut r = TestResult::from_bool(false);
        r.err = Some(msg.into());
        r
    }

    /// Produces a test result that instructs `quickcheck` to ignore it.
    /// This is useful for restricting the domain of your properties.
    /// When a test is discarded, `quickcheck` will replace it with a
    /// fresh one (up to a certain limit).
    pub fn discard() -> TestResult {
        TestResult {
            status: Discard,
            arguments: None,
            err: None,
            generation_time: std::time::Duration::default(),
            execution_time: std::time::Duration::default(),
            shrinking_time: std::time::Duration::default(),
        }
    }

    /// Converts a `bool` to a `TestResult`. A `true` value indicates that
    /// the test has passed and a `false` value indicates that the test
    /// has failed.
    pub fn from_bool(b: bool) -> TestResult {
        TestResult {
            status: if b { Pass } else { Fail },
            arguments: None,
            err: None,
            generation_time: std::time::Duration::default(),
            execution_time: std::time::Duration::default(),
            shrinking_time: std::time::Duration::default(),
        }
    }

    /// Converts an `Option<bool>` to a `TestResult`. A `Some(true)` value
    /// indicates that the test has passed, a `Some(false)` value indicates
    /// that the test has failed, and a `None` value indicates that the test
    /// has been discarded.
    pub fn from_option_bool(b: Option<bool>) -> TestResult {
        match b {
            Some(true) => TestResult::from_bool(true),
            Some(false) => TestResult::from_bool(false),
            None => TestResult::discard(),
        }
    }

    /// Tests if a "procedure" fails when executed. The test passes only if
    /// `f` generates a task failure during its execution.
    pub fn must_fail<T, F>(f: F) -> TestResult
    where
        F: FnOnce() -> T,
        F: 'static,
        T: 'static,
    {
        let f = panic::AssertUnwindSafe(f);
        TestResult::from_bool(panic::catch_unwind(f).is_err())
    }

    /// Returns `true` if and only if this test result describes a failing
    /// test.
    pub fn is_failure(&self) -> bool {
        match self.status {
            Fail => true,
            Pass | Discard => false,
        }
    }

    /// Returns `true` if and only if this test result describes a failing
    /// test as a result of a run time error.
    pub fn is_error(&self) -> bool {
        self.is_failure() && self.err.is_some()
    }

    fn failed_msg(&self) -> String {
        let arguments_msg = match self.arguments {
            None => "No Arguments Provided".to_owned(),
            Some(ref args) => format!("Arguments: ({})", args.join(", ")),
        };
        match self.err {
            None => format!("[quickcheck] TEST FAILED. {arguments_msg}"),
            Some(ref err) => format!(
                "[quickcheck] TEST FAILED (runtime error). {arguments_msg}\nError: {err}"
            ),
        }
    }
}

impl From<bool> for TestResult {
    /// A shorter way of producing a `TestResult` from a `bool`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use quickcheck::TestResult;
    /// let result: TestResult = (2 > 1).into();
    /// assert_eq!(result, TestResult::passed());
    /// ```
    fn from(b: bool) -> TestResult {
        TestResult::from_bool(b)
    }
}

/// `Testable` describes types (e.g., a function) whose values can be
/// tested.
///
/// Anything that can be tested must be capable of producing a `TestResult`
/// given a random number generator. This is trivial for types like `bool`,
/// which are just converted to either a passing or failing test result.
///
/// For functions, an implementation must generate random arguments
/// and potentially shrink those arguments if they produce a failure.
///
/// It's unlikely that you'll have to implement this trait yourself.
pub trait Testable: 'static {
    fn result(&self, _: &mut Gen) -> TestResult;
}

impl Testable for bool {
    fn result(&self, _: &mut Gen) -> TestResult {
        TestResult::from_bool(*self)
    }
}

impl Testable for Option<bool> {
    fn result(&self, _: &mut Gen) -> TestResult {
        TestResult::from_option_bool(*self)
    }
}

impl Testable for () {
    fn result(&self, _: &mut Gen) -> TestResult {
        TestResult::passed()
    }
}

impl Testable for TestResult {
    fn result(&self, _: &mut Gen) -> TestResult {
        self.clone()
    }
}

impl<A, E> Testable for Result<A, E>
where
    A: Testable,
    E: Debug + 'static,
{
    fn result(&self, g: &mut Gen) -> TestResult {
        match *self {
            Ok(ref r) => r.result(g),
            Err(ref err) => TestResult::error(format!("{err:?}")),
        }
    }
}

#[cfg(feature = "etna")]
/// Return a vector of the debug formatting of each item in `args`
fn debug_reprs(args: &[&dyn Serialize]) -> Vec<String> {
    args.iter()
        .map(|x| format!("{}", serde_sexpr::to_string(x).unwrap()))
        .collect()
}

#[cfg(not(feature = "etna"))]
/// Return a vector of the debug formatting of each item in `args`
fn debug_reprs(args: &[&dyn Debug]) -> Vec<String> {
    args.iter().map(|x| format!("{x:?}")).collect()
}

#[cfg(feature = "etna")]
pub trait Sample {
    /// For an Function `fn(T1, T2) -> T`, this method samples
    /// a pair of arguments `(T1, T2)` and returns the elapsed time
    /// it took to sample them, along with the function itself.
    fn sample(&self, g: &mut Gen) -> (Duration, Vec<String>);
}

#[cfg(feature = "etna")]
macro_rules! sampling_fn {
    ($($name: ident),*) => {

impl<T,
     $($name: Arbitrary + Serialize),*> Sample for fn($($name),*) -> T {
    #[allow(non_snake_case)]
    fn sample(&self, g: &mut Gen) -> (Duration, Vec<String>) {
        let t0 = std::time::Instant::now();
        let a: ($($name,)*) = Arbitrary::arbitrary(g);
        let generation_time = t0.elapsed();

        let ($(ref $name,)*) : ($($name,)*) = a;
        let arguments = debug_reprs(&[$($name),*]);
        (generation_time, arguments)
    }
}}}

#[cfg(feature = "etna")]
sampling_fn!();
#[cfg(feature = "etna")]
sampling_fn!(A);
#[cfg(feature = "etna")]
sampling_fn!(A, B);
#[cfg(feature = "etna")]
sampling_fn!(A, B, C);
#[cfg(feature = "etna")]
sampling_fn!(A, B, C, D);
#[cfg(feature = "etna")]
sampling_fn!(A, B, C, D, E);
#[cfg(feature = "etna")]
sampling_fn!(A, B, C, D, E, F);
#[cfg(feature = "etna")]
sampling_fn!(A, B, C, D, E, F, G);
#[cfg(feature = "etna")]
sampling_fn!(A, B, C, D, E, F, G, H);

#[cfg(feature = "etna")]
macro_rules! testable_fn {
    ($($name: ident),*) => {

impl<T: Testable,
     $($name: Arbitrary + Debug + Serialize),*> Testable for fn($($name),*) -> T {
    #[allow(non_snake_case)]
    fn result(&self, g: &mut Gen) -> TestResult {
        fn shrink_failure<T: Testable, $($name: Arbitrary + Debug + Serialize),*>(
            g: &mut Gen,
            self_: fn($($name),*) -> T,
            a: ($($name,)*),
        ) -> Option<TestResult> {
            for t in a.shrink() {
                let ($($name,)*) = t.clone();
                let mut r_new = safe(move || {self_($($name),*)}).result(g);
                if r_new.is_failure() {
                    {
                        let ($(ref $name,)*) : ($($name,)*) = t;
                        r_new.arguments = Some(debug_reprs(&[$($name),*]));
                    }

                    // The shrunk value *does* witness a failure, so keep
                    // trying to shrink it.
                    let shrunk = shrink_failure(g, self_, t);

                    // If we couldn't witness a failure on any shrunk value,
                    // then return the failure we already have.
                    return Some(shrunk.unwrap_or(r_new))
                }
            }
            None
        }


        let self_ = *self;
        let t0 = std::time::Instant::now();
        let a: ($($name,)*) = Arbitrary::arbitrary(g);
        let generation_time = t0.elapsed();

        let ( $($name,)* ) = a.clone();
        let t0 = std::time::Instant::now();
        let mut r = safe(move || {self_($($name),*)}).result(g);
        let execution_time = t0.elapsed();

        let ($(ref $name,)*) : ($($name,)*) = a;
        r.arguments = Some(debug_reprs(&[$($name),*]));

        match r.status {
            Pass|Discard => r,
            Fail => {
                let t0 = std::time::Instant::now();
                let mut r = shrink_failure(g, self_, a).unwrap_or(r);
                let shrinking_time = t0.elapsed();

                r.generation_time = generation_time;
                r.execution_time = execution_time;
                r.shrinking_time = shrinking_time;

                r
            }
        }
    }
}}}

#[cfg(not(feature = "etna"))]
macro_rules! testable_fn {
    ($($name: ident),*) => {

impl<T: Testable,
     $($name: Arbitrary + Debug),*> Testable for fn($($name),*) -> T {
    #[allow(non_snake_case)]
    fn result(&self, g: &mut Gen) -> TestResult {
        fn shrink_failure<T: Testable, $($name: Arbitrary + Debug),*>(
            g: &mut Gen,
            self_: fn($($name),*) -> T,
            a: ($($name,)*),
        ) -> Option<TestResult> {
            for t in a.shrink() {
                let ($($name,)*) = t.clone();
                let mut r_new = safe(move || {self_($($name),*)}).result(g);
                if r_new.is_failure() {
                    {
                        let ($(ref $name,)*) : ($($name,)*) = t;
                        r_new.arguments = Some(debug_reprs(&[$($name),*]));
                    }

                    // The shrunk value *does* witness a failure, so keep
                    // trying to shrink it.
                    let shrunk = shrink_failure(g, self_, t);

                    // If we couldn't witness a failure on any shrunk value,
                    // then return the failure we already have.
                    return Some(shrunk.unwrap_or(r_new))
                }
            }
            None
        }

        let self_ = *self;
        let a: ($($name,)*) = Arbitrary::arbitrary(g);
        let ( $($name,)* ) = a.clone();
        let r = safe(move || {self_($($name),*)}).result(g);
        match r.status {
            Pass|Discard => r,
            Fail => {
                shrink_failure(g, self_, a).unwrap_or(r)
            }
        }
    }
}}}

testable_fn!();
testable_fn!(A);
testable_fn!(A, B);
testable_fn!(A, B, C);
testable_fn!(A, B, C, D);
testable_fn!(A, B, C, D, E);
testable_fn!(A, B, C, D, E, F);
testable_fn!(A, B, C, D, E, F, G);
testable_fn!(A, B, C, D, E, F, G, H);

fn safe<T, F>(fun: F) -> Result<T, String>
where
    F: FnOnce() -> T,
    F: 'static,
    T: 'static,
{
    panic::catch_unwind(panic::AssertUnwindSafe(fun)).map_err(|any_err| {
        // Extract common types of panic payload:
        // panic and assert produce &str or String
        if let Some(&s) = any_err.downcast_ref::<&str>() {
            s.to_owned()
        } else if let Some(s) = any_err.downcast_ref::<String>() {
            s.to_owned()
        } else {
            "UNABLE TO SHOW RESULT OF PANIC.".to_owned()
        }
    })
}

#[cfg(test)]
mod test {
    use crate::{Gen, QuickCheck};

    #[test]
    fn shrinking_regression_issue_126() {
        fn thetest(vals: Vec<bool>) -> bool {
            vals.iter().filter(|&v| *v).count() < 2
        }
        let failing_case = QuickCheck::new()
            .quicktest(thetest as fn(vals: Vec<bool>) -> bool)
            .unwrap_err();
        #[cfg(not(feature = "etna"))]
        let expected_argument = format!("{:?}", [true, true]);
        #[cfg(feature = "etna")]
        let expected_argument = r#"(true true)"#.to_owned();
        assert_eq!(failing_case.arguments, Some(vec![expected_argument]));
    }

    #[test]
    fn size_for_small_types_issue_143() {
        fn t(_: i8) -> bool {
            true
        }
        QuickCheck::new()
            .set_rng(Gen::new(129))
            .quickcheck(t as fn(i8) -> bool);
    }

    #[test]
    fn regression_signed_shrinker_panic() {
        fn foo_can_shrink(v: i8) -> bool {
            let _ = crate::Arbitrary::shrink(&v).take(100).count();
            true
        }
        crate::quickcheck(foo_can_shrink as fn(i8) -> bool);
    }
}
