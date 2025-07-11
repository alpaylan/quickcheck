use quickcheck::quickcheck;

fn reverse<T: Clone>(xs: &[T]) -> Vec<T> {
    let mut rev = vec![];
    if xs.is_empty() {
        return rev;
    }
    for x in xs[1..].iter() {
        rev.insert(0, x.clone());
    }
    rev
}

fn main() {
    fn equality_after_applying_twice(xs: Vec<isize>) -> bool {
        xs == reverse(&reverse(&xs))
    }
    quickcheck(equality_after_applying_twice as fn(Vec<isize>) -> bool);
}
