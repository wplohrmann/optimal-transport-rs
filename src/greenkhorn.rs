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
}

impl SinkhornProjection
{
    pub fn new(cost: &ArrayView2<f32>, reg: f32) -> Self
    {
        let mut p = cost.mapv(|x| (-x / reg).exp());
        p /= p.sum();
        let row_sum = p.sum_axis(Axis(1));
        let col_sum = p.sum_axis(Axis(0));
        SinkhornProjection{p, row_sum, col_sum}
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
}

struct ProjectionDistance
{
    row_rho: Array1<f32>,
    col_rho: Array1<f32>,
    row_distances: Array1<f32>,
    col_distances: Array1<f32>
}

impl ProjectionDistance
{
    pub fn new(projection: &SinkhornProjection, r: &ArrayView1<f32>, c: &ArrayView1<f32>) -> Self
    {
        let row_rho = Zip::from(&projection.row_sum).and(r).apply_collect(|a, b| rho(*a, *b));
        let row_distances = Zip::from(&projection.row_sum).and(r).apply_collect(|a, b| (a-b).abs());

        let col_rho = Zip::from(&projection.col_sum).and(c).apply_collect(|a, b| rho(*a, *b));
        let col_distances = Zip::from(&projection.col_sum).and(c).apply_collect(|a, b| (a-b).abs());

        ProjectionDistance{row_rho, col_rho, row_distances, col_distances}
    }

    pub fn eval(&self) -> f32
    {
        self.row_distances.sum() + self.col_distances.sum()
    }

    pub fn update_row(&mut self, row_index: usize, r: &ArrayView1<f32>)
    {
    }

    pub fn update_col(&mut self, col_index: usize, c: &ArrayView1<f32>)
    {
    }
}

pub fn greenkhorn(r: &ArrayView1< f32 >, c: &ArrayView1<f32>, cost: &ArrayView2< f32 >, reg: f32) -> Array2< f32 >
{
    let eps = 1e-8;
    let mut solution = SinkhornProjection::new(cost, reg);
    let mut solution_distance = ProjectionDistance::new(&solution, r, c);

    while solution_distance.eval() > eps
    {
        let max_row = (0, 1.); // solution_distance.max_row();
        let max_col = (0, 1.); // solution_distance.max_col();
        if max_row.1 > max_col.1
        {
            solution.update_row(max_row.0, r);
            solution_distance.update_row(max_row.0, r);
        }
        else
        {
            solution.update_col(max_col.0, c);
            solution_distance.update_col(max_col.0, c);
        }
    }
    solution.p
}
