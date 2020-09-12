use std::cmp;
use ndarray::{Array, Array1, Array2, s};

// use num_traits::Float;

// fn simplex_solver<T: Num>(Array2<T>: table) -> (Array2, T) {

// }

// for now index is position in these histograms
pub fn ot_to_simplex(from_histogram: Array1<f32>, to_histogram: Array1<f32>) -> Array2<f32> {
    let m: usize = from_histogram.len();
    let n: usize = to_histogram.len();
    let num_rows: usize = 1 + m + n;
    let num_cols: usize = 2 + (m * n);
    let mut flow_matrix: Array2<f32> = Array::zeros((num_rows, num_cols));

    flow_matrix[[0, 0]] = 1.;

    println!("{}", ((1 - 2) as f32).abs());

    for i in 1..m + 1 {
        // objective function
        for j in 1..n + 1 {
            flow_matrix[[0, ((i - 1) * n) + j]] = (cmp::max(j, i) - cmp::min(j, i)) as f32;
        }

        // from_histogram constraints
        flow_matrix.slice_mut(s![i, 1 + ((i - 1) * n) .. (1 + (i * n))]).fill(1.);
        flow_matrix[[i, num_cols - 1]] = from_histogram[i - 1];
    }

    // to_histogram constraints
    for j in 0..n {
        for i in 0..m {
            flow_matrix[[1 + m + j, j + 1 + (i * n)]] = 1.;
        }
        flow_matrix[[1 + m + j, num_cols - 1]] = to_histogram[j];
    }
    flow_matrix
}