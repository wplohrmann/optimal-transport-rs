use std::cmp;
use ndarray::{Array, Array1, Array2, s, Zip, array};
use ndarray_stats::QuantileExt;

pub fn solve_simplex_tableau(tableau: &mut Array2<f32>) -> &Array2<f32> {
    // expects a tableau with some negative values in the objective (last) row

    while *tableau.slice(s![tableau.shape()[0] - 1, ..]).min().unwrap() < 0. {
        // Find the pivot column corresponding the lowest value in first row
        let pivot_column_idx = tableau.slice(s![tableau.shape()[0] - 1, ..tableau.shape()[1] - 1]).argmin().unwrap();

        // Divide each element in the rightmost column by the corresponding element in pivot column to find a ratio
        let right_hand_column = tableau.slice(s![..tableau.shape()[0] - 1, tableau.shape()[1] - 1]).to_owned();
        let right_hand_column_divided = right_hand_column / tableau.slice(s![..tableau.shape()[0] - 1, pivot_column_idx]);

        // Find the pivot row corresponding to the minimum ratio found in previous step, this corresponds to the tightest constraint
        let pivot_row_idx = right_hand_column_divided.argmin_skipnan().unwrap();

        // Divide pivot row by the pivot element
        let pivoted_row = tableau.slice(s![pivot_row_idx, ..]).mapv(|a| a / tableau[[pivot_row_idx, pivot_column_idx]]);
        tableau.slice_mut(s![pivot_row_idx, ..]).assign(&pivoted_row);

        // For all other rows including the last row for the objective, subtract a multiple of the pivot row such that the pivot column element becomes zero
        for (i, row) in tableau.outer_iter_mut().enumerate() {
            if i != pivot_row_idx {
                let row_minus_factor = row[pivot_column_idx];
                Zip::from(row)
                .and(&pivoted_row)
                .apply(|a, &b| {
                    *a -= row_minus_factor * b;
                });
            }
        }
    } // Repeat the steps above until all elements in the last row are non-negative

    tableau
}

// for now index is position in these histograms
pub fn ot_to_simplex_tableau(from_histogram: Array1<f32>, to_histogram: Array1<f32>) -> Array2<f32> {
    let m: usize = from_histogram.len();
    let n: usize = to_histogram.len();
    let num_rows: usize = 1 + m + n;
    let num_cols: usize = 2 + (m * n);
    let mut flow_matrix: Array2<f32> = Array::zeros((num_rows, num_cols));

    flow_matrix[[num_rows - 1, 0]] = 1.;

    println!("{}", ((1 - 2) as f32).abs());

    for i in 0..m {
        // objective function to minimise placed in last row
        for j in 0..n {
            flow_matrix[[num_rows - 1, (i * n) + j + 1]] = (cmp::max(j, i) - cmp::min(j, i)) as f32;
        }

        // from_histogram constraints
        flow_matrix.slice_mut(s![i, 1 + (i * n) .. 1 + ((i + 1) * n)]).fill(1.);
        flow_matrix[[i, num_cols - 1]] = from_histogram[i];
    }

    // to_histogram constraints
    for j in 0..n {
        for i in 0..m {
            flow_matrix[[m + j, 1 + j + (i * n)]] = 1.;
        }
        flow_matrix[[m + j, num_cols - 1]] = to_histogram[j];
    }

    // must transpose to convert minimisation problem to maximisation problem
    flow_matrix.reversed_axes()
    // TODO: figure out how this tableau should be handled by simplex given that
    // it has no negative variables in the objective row to start with
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_solved() {
        let mut test_tableau = array![[2., 1., 1., 0., 0., 3.],
                                      [1., 2., 0., 1., 0., 9.],
                                      [-8., -8., 0., 0., 1., 0.]];

        let solved_tableau = array![[2., 1., 1., 0., 0., 3.],
                                    [-3., 0., -2., 1., 0., 3.],
                                    [8., 0., 8., 0., 1., 24.]];

        assert_eq!(*solve_simplex_tableau(&mut test_tableau), solved_tableau);
    }
}