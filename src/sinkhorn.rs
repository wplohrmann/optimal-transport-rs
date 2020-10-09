use ndarray::{Array1, Array2, Axis, ArrayView1, ArrayView2};

pub fn sinkhorn(a: &ArrayView1< f32 >, b: &ArrayView1< f32 >, cost: &ArrayView2< f32 >, reg: f32) -> Array2< f32 >
{
    let n = cost.nrows();
    let m = cost.ncols();

    let mut p = cost.mapv( |x| (-x / reg).exp());

    let mut u = Array1::zeros(n);
    while (u - p.sum_axis(Axis(1))).mapv(|x: f32| x.powi(2)).sum() > 1e-3
    {
        u = p.sum_axis(Axis(1));
        for i in 0..n {
            for j in 0..m {
                p[[i, j]] *= a[i] / u[i];
            }
        }
        let v = p.sum_axis(Axis(0));
        for i in 0..n {
            for j in 0..m {
                p[[i, j]] *= b[j] / v[j];
            }
        }
    }
    p
}
