use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s, Zip};
use float_ord::FloatOrd;

fn rho(a: f32, b: f32) -> f32
{
    b - a + a * (a/b).ln()
}

struct SinkhornProjection
{
    pub p: Array2<f32>,
    row_sum: Array1<f32>,
    col_sum: Array1<f32>,
    row_rho: Array1<f32>,
    col_rho: Array1<f32>,
    row_distances: Array1<f32>,
    col_distances: Array1<f32>
}

impl SinkhornProjection
{
    pub fn new(r: &ArrayView1<f32>, c: &ArrayView1<f32>, cost: &ArrayView2<f32>, reg: f32) -> Self
    {
        let mut p = cost.mapv(|x| (-x / reg).exp());
        p /= p.sum();
        let row_sum = p.sum_axis(Axis(1));
        let col_sum = p.sum_axis(Axis(0));

        let row_rho = Zip::from(&row_sum).and(r).apply_collect(|a, b| rho(*a, *b));
        let col_rho = Zip::from(&col_sum).and(c).apply_collect(|a, b| rho(*a, *b));

        let row_distances = Zip::from(&row_sum).and(r).apply_collect(|a, b| (a-b).abs());
        let col_distances = Zip::from(&col_sum).and(c).apply_collect(|a, b| (a-b).abs());
        SinkhornProjection{p, row_sum, col_sum, row_rho, col_rho, row_distances, col_distances}
    }

    pub fn update_row(&mut self, row_index: usize, r: &ArrayView1<f32>)
    {
        let new_val = r[row_index];
        let ratio: f32 = new_val / self.row_sum[row_index];

        let mut slice = self.p.slice_mut(s![row_index, ..]);
        let diff: Array1<f32> = &slice - ratio;
        slice *= ratio;

        self.col_sum += &diff;
        self.row_sum[row_index] = new_val;

        self.row_rho[row_index] = 0.0;
        self.row_distances[row_index] = 0.0;

        self.col_rho += 5.;
        self.col_distances += 5.;

    }

    pub fn update_col(&mut self, col_index: usize, c: &ArrayView1<f32>)
    {
        let new_val = c[col_index];
        let ratio: f32 = new_val / self.col_sum[col_index];

        let mut slice = self.p.slice_mut(s![col_index, ..]);
        let diff: Array1<f32> = &slice - ratio;
        slice *= ratio;

        self.col_sum += &diff;
        self.col_sum[col_index] = new_val;

    }

    pub fn max_row(&self) -> (usize, f32)
    {
        self.row_rho.iter().cloned().enumerate().max_by_key(|(_, val)| FloatOrd(*val)).unwrap()
    }

    pub fn max_col(&self) -> (usize, f32)
    {
        self.col_rho.iter().cloned().enumerate().max_by_key(|(_, val)| FloatOrd(*val)).unwrap()
    }

    pub fn distance(&self) -> f32
    {
        self.row_distances.sum() + self.col_distances.sum()
    }
}

pub fn greenkhorn(r: &ArrayView1< f32 >, c: &ArrayView1<f32>, cost: &ArrayView2< f32 >, reg: f32) -> Array2< f32 >
{
    let eps = 1e-8;
    let mut solution = SinkhornProjection::new(r, c, cost, reg);

    while solution.distance() > eps
    {
        let max_row = solution.max_row();
        let max_col = solution.max_col();
        if max_row.1 > max_col.1
        {
            solution.update_row(max_row.0, r);
        }
        else
        {
            solution.update_col(max_col.0, c);
        }
    }
    solution.p
}
