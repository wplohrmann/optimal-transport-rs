use ndarray::{Array1, Array2, Axis};
use float_ord::FloatOrd;

pub fn sinkhorn(a: Array1< f32 >, b: Array1< f32 >, cost: Array2< f32 >, reg: f32) -> Array2< f32 >
{
    let n = cost.nrows();
    let m = cost.ncols();
    let epsilon = FloatOrd(1e-8);

    let mut P = cost.mapv( |x| (-x / reg).exp());
    P /= P.sum();

    let mut u = Array1::zeros(n);
    while (u -P.sum_axis(Axis(1))).mapv( |x: f32| FloatOrd(x.abs())).iter().max().unwrap() > &epsilon
    {
        u = P.sum_axis(Axis(1));
        for i in 0..n {
            for j in 0..m {
                P[[i, j]] *= a[i] / u[i];
            }
        }
        let v = P.sum_axis(Axis(0));
        for i in 0..n {
            for j in 0..m {
                P[[i, j]] *= b[i] / v[i];
            }
        }
    }
    P
}
