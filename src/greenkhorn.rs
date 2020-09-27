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
    let rho = |a: &f32, b: &f32| b - a + a * (a/b).log2();

    let mut row_rho = solution.distance_row(r, rho);
    let mut col_rho = solution.distance_col(c, rho);

    let mut row_distances = solution.distance_row(r, abs);
    let mut col_distances = solution.distance_col(c, abs);

    while row_distances.sum() + col_distances.sum() > 2.
    // for _ in 0..10000
    {
        dbg!(row_distances.sum() + col_distances.sum());
        let max_row = row_rho.iter().cloned().enumerate().max_by_key(|(_, val)| FloatOrd(*val)).unwrap();
        let max_col = col_rho.iter().cloned().enumerate().max_by_key(|(_, val)| FloatOrd(*val)).unwrap();
        if max_row.1 > max_col.1
        {
            let index = max_row.0;
            solution.update_row(index, r);
            row_rho[index] = 0.0;
            row_distances[index] = 0.0;
            col_rho = solution.distance_col(c, rho);
            col_distances = solution.distance_col(c, abs);
        }
        else
        {
            let index = max_col.0;
            solution.update_col(index, c);
            col_rho[index] = 0.0;
            col_distances[index] = 0.0;
            row_rho = solution.distance_row(r, rho);
            row_distances = solution.distance_row(r, abs);
        }
    }
    solution.p
}

#[cfg(test)]
mod tests
{
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn it_works(a: f32, b: f32)
    {
        assert_eq!(b+a, a+b);
    }
}
