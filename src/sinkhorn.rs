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
    while (u.clone()-P.sum_axis(Axis(1))).mapv( |x: f32| FloatOrd(x.abs())).iter().max().unwrap() > &epsilon
    {
        u = P.sum_axis(Axis(1));
        P = P * (a.clone() / u.clone()).broadcast((n, m)).unwrap();
        P = P * (b.clone() / P.sum_axis(Axis(0))).broadcast((n, m)).unwrap();
    }
    P
}
