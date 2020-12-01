use numpy::{PyArray1, PyArray2};
use numpy::convert::IntoPyArray;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod sinkhorn;
mod lp_solver;

use sinkhorn::sinkhorn as impl_sinkhorn;
use lp_solver::calculate_1D_ot as impl_calculate_1D_ot;

#[pyfunction]
fn calculate_1D_ot(py: Python<'_>, a: &PyArray1<i32>, b: &PyArray1<i32>, cost: &PyArray2<i32>) -> PyResult<(i32, Py<PyArray2<u32>>)> {
   let (cost, transport_plan) = impl_calculate_1D_ot(&a.to_owned_array(), &b.to_owned_array(), &cost.to_owned_array());
   Ok((cost, transport_plan.into_pyarray(py).to_owned()))
}

#[pyfunction]
fn sinkhorn(py: Python<'_>, a: &PyArray1<f32>, b: &PyArray1<f32>, cost: &PyArray2<f32>, reg: f32, maybe_warm_u: Option<&PyArray1<f32>>) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray1<f32>>)> {
    let warm_u_rs = if let Some(warm_u) = maybe_warm_u {
       Some(warm_u.to_owned_array())
    }
    else{
       None
    };
   let (transport_plan, u) = impl_sinkhorn(a.to_owned_array(), b.to_owned_array(), cost.to_owned_array(), reg, warm_u_rs);
   Ok((transport_plan.into_pyarray(py).to_owned(), u.into_pyarray(py).to_owned()))
}

#[pymodule]
fn rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sinkhorn))?;
    m.add_wrapped(wrap_pyfunction!(calculate_1D_ot))?;

    Ok(())
}
