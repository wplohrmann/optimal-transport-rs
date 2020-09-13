mod simplex;

use ndarray::{array};

fn main() {
    println!("Hello, simplex!");

    let from_histogram = array![2., 1.5];
    let to_histogram = array![0.5, 3.];

    let mut simplex_tableau = simplex::ot_to_simplex_tableau(from_histogram, to_histogram);

    println!("simplex tableau:\n{}", simplex_tableau);

    let solved_tableau = simplex::solve_simplex_tableau(&mut simplex_tableau);

    println!("solved tableau:\n{}", solved_tableau);

    let mut test_tableau = array![[2., 1., 1., 0., 0., 3.],
                                  [1., 2., 0., 1., 0., 9.],
                                  [-8., -8., 0., 0., 1., 0.]];

    println!("test tableau:\n{}", test_tableau);

    let tested_tableau = simplex::solve_simplex_tableau(&mut test_tableau);

    println!("tested tableau:\n{}", tested_tableau);
}
