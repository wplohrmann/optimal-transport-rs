mod simplex;

use ndarray::{array};

fn main() {
    println!("Hello, world!");

    let from_histogram = array![1., 2., 3.];
    let to_histogram = array![2., 2., 2.];

    let simplex_tableau = simplex::ot_to_simplex(from_histogram, to_histogram);

    println!("{}", simplex_tableau);
}
