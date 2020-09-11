use ndarray::{Array1, Array2, Axis};
use float_ord::FloatOrd;

pub fn sinkhorn(a: Array1< f32 >, b: Array1< f32 >, cost: Array2< f32 >, reg: f32) -> Array2< f32 >
{
    let n = cost.nrows();
    let m = cost.ncols();
    let epsilon = FloatOrd(1e-8);

    let mut p = cost.mapv( |x| (-x / reg).exp());

    let mut u = Array1::zeros(n);
    while (u - p.sum_axis(Axis(1))).mapv( |x: f32| FloatOrd(x.abs())).iter().max().unwrap() > &epsilon
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
