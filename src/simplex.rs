use ndarray::{Array, Array1, Array2, s};

use num_traits::Float;

// fn simplex_solver<T: Num>(Array2<T>: table) -> (Array2, T) {

// }

// for now index is position in these histograms
fn ot_to_simplex<T: Float + std::convert::From<usize>>(from_histogram: Array1<T>, to_histogram: Array1<T>) -> Array2<T> {
    let m: usize = from_histogram.len();
    let n: usize = to_histogram.len();
    let num_rows: usize = 1 + m + n;
    let num_cols: usize = 1 + ((1 + m) * (1 + n));
    let mut flow_matrix: Array2<T> = Array::zeros((num_rows, num_cols));

    flow_matrix[[0, 0]] = 1.into();

    for i in 1..m {
        for j in 1..n {
            flow_matrix[[0, ((i - 1) * m) + j]] = (j.into() - i.into()).abs();
        }
        
        for col in 1..num_cols {
        }
        let active_slice = flow_matrix.slice_mut(s![[i, 1 + ((i - 1) * n): 1 + (i * n)]]);
        active_slice.fill(1.into());
    }

    flow_matrix
}