use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s, Zip};
use float_ord::FloatOrd;


struct SinkhornProjection
{
    pub p: Array2<f32>,
    row_sum: Row<Array1<f32>>,
    col_sum: Col<Array1<f32>>,
}

impl SinkhornProjection
{
    pub fn new(cost: &ArrayView2<f32>, reg: f32) -> Self
    {
        let mut p = cost.mapv(|x| (-x / reg).exp());
        p /= p.sum();
        let row_sum = Row(p.sum_axis(Axis(1)));
        let col_sum = Col(p.sum_axis(Axis(0)));

        SinkhornProjection{p, row_sum, col_sum}
    }

    pub fn update_row(&mut self, row_index: usize, r: &Row<ArrayView1<f32>>)
    {
        let new_val = r.0[row_index];
        let ratio: f32 = new_val / self.row_sum.0[row_index];

        let mut slice = self.p.slice_mut(s![row_index, ..]);
        let diff: Array1<f32> = ratio - &slice;
        slice *= ratio;

        self.col_sum.0 += &diff;
        self.row_sum.0[row_index] = new_val;

        self.col_sum.0 = self.p.sum_axis(Axis(0));
        self.row_sum.0 = self.p.sum_axis(Axis(1));
        // debug_assert_eq!(self.col_sum.0, self.p.sum_axis(Axis(0)));
        // debug_assert_eq!(self.row_sum.0, self.p.sum_axis(Axis(1)));

    }

    pub fn update_col(&mut self, col_index: usize, c: &Col<ArrayView1<f32>>)
    {
        let new_val = c.0[col_index];
        let ratio: f32 = new_val / self.col_sum.0[col_index];

        let mut slice = self.p.slice_mut(s![.., col_index]);
        let diff: Array1<f32> = ratio - &slice;
        slice *= ratio;

        self.row_sum.0 += &diff;
        self.col_sum.0[col_index] = new_val;

        // debug_assert_eq!(self.col_sum.0, self.p.sum_axis(Axis(0)));
        // debug_assert_eq!(self.row_sum.0, self.p.sum_axis(Axis(1)));
        self.col_sum.0 = self.p.sum_axis(Axis(0));
        self.row_sum.0 = self.p.sum_axis(Axis(1));

    }

    pub fn distance_row(&self, row: &Row<ArrayView1<f32>>, distance_func: impl Fn(&f32, &f32) -> f32) -> Array1<f32>
    {
        Zip::from(&row.0).and(&self.row_sum.0).apply_collect(distance_func)
    }

    pub fn distance_col(&self, col: &Col<ArrayView1<f32>>, distance_func: impl Fn(&f32, &f32) -> f32) -> Array1<f32>
    {
        Zip::from(&col.0).and(&self.col_sum.0).apply_collect(distance_func)
    }
}

pub struct Row<T> (pub T);
pub struct Col<T> (pub T);

pub fn greenkhorn(r: &Row<ArrayView1< f32 >>, c: &Col<ArrayView1<f32>>, cost: &ArrayView2< f32 >, reg: f32) -> Array2< f32 >
{
    let mut solution = SinkhornProjection::new(cost, reg);
    let abs = |a: &f32, b: &f32| (a-b).abs();
    let rho = |a: &f32, b: &f32| (b - a + a * (a/b).log2()).abs();

    let mut row_rho = solution.distance_row(r, rho);
    let mut col_rho = solution.distance_col(c, rho);

    let mut row_distances = solution.distance_row(r, abs);
    let mut col_distances = solution.distance_col(c, abs);

    while (row_distances.sum() + col_distances.sum()) > 1e-3
    {
        let max_row = row_rho.iter().cloned().enumerate().max_by_key(|(_, val)| FloatOrd(*val)).unwrap();
        let max_col = col_rho.iter().cloned().enumerate().max_by_key(|(_, val)| FloatOrd(*val)).unwrap();
        if max_row.1 > max_col.1
        {
            let index = max_row.0;
            solution.update_row(index, r);
            row_rho = solution.distance_row(r, rho);
            col_rho = solution.distance_col(c, rho);

            row_distances = solution.distance_row(r, abs);
            col_distances = solution.distance_col(c, abs);
        }
        else
        {
            let index = max_col.0;
            solution.update_col(index, c);
            row_rho = solution.distance_row(r, rho);
            col_rho = solution.distance_col(c, rho);

            row_distances = solution.distance_row(r, abs);
            col_distances = solution.distance_col(c, abs);
        }
    }
    solution.p
}

#[cfg(test)]
mod tests
{
    use quickcheck_macros::quickcheck;
    use ndarray::Array2;
    use super::*;

    #[quickcheck]
    fn it_works(row: Vec<f32>, col: Vec<f32>)
    {
        if row.len() > 0 && col.len() > 0 && row.iter().sum::<f32>() > 0_f32 && col.iter().sum::<f32>() > 0_f32
        {
            let row_raw = {let x = Array1::from(row.iter().copied().map(f32::abs).collect::<Vec<f32>>()); &x / x.sum()};
            let col_raw = {let x = Array1::from(col.iter().copied().map(f32::abs).collect::<Vec<f32>>()); &x / x.sum()};
            let mut cost = Array2::zeros((row_raw.len(), col_raw.len()));
            for i in 0..row.len()
            {
                for j in 0..col.len()
                {
                    cost[[i,j]] = (i as f32 -j as f32).abs();
                }
            }
            let reg = 0.01;
            let mut solution = SinkhornProjection::new(&cost.view(), reg);
            assert_eq!(solution.row_sum.0.len(), row_raw.len());
            assert_eq!(solution.col_sum.0.len(), col_raw.len());

            solution.update_row(0, &Row(row_raw.view()));
            solution.update_col(0, &Col(col_raw.view()));
        }
    }
    #[quickcheck]
    fn rho_is_nonnegative(a: f32, b: f32)
    {
        if a > 0. && b > 0. && a <= 1. && b <= 1.
        {
            let rho = b - a + a * (a/b).log2();
            dbg!(rho);
            assert!(rho >= 0.);
        }
    }
}
