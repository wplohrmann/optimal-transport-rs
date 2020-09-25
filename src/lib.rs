use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray2};
use numpy::convert::IntoPyArray;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod sinkhorn;
mod greenkhorn;
mod lp_solver;

use sinkhorn::sinkhorn as impl_sinkhorn;
use greenkhorn::greenkhorn as impl_greenkhorn;
use lp_solver::calculate_1D_ot as impl_calculate_1D_ot;

#[pyfunction]
fn calculate_1D_ot(py: Python<'_>, a: PyReadonlyArray1<i32>, b: PyReadonlyArray1<i32>, cost: PyReadonlyArray2<i32>) -> PyResult<(i32, Py<PyArray2<u32>>)> {
   let (cost, transport_plan) = impl_calculate_1D_ot(&a.as_array(), &b.as_array(), &cost.as_array());
   Ok((cost, transport_plan.into_pyarray(py).to_owned()))
}

#[pyfunction]
fn sinkhorn(py: Python<'_>, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>, cost: PyReadonlyArray2<f32>, reg: f32) -> PyResult<Py<PyArray2<f32>>> {
   let transport_plan = impl_sinkhorn(&a.as_array(), &b.as_array(), &cost.as_array(), reg);
   Ok(transport_plan.into_pyarray(py).to_owned())
}

#[pyfunction]
fn greenkhorn(py: Python<'_>, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>, cost: PyReadonlyArray2<f32>, reg: f32) -> PyResult<Py<PyArray2<f32>>> {
   let transport_plan = impl_greenkhorn(&a.as_array(), &b.as_array(), &cost.as_array(), reg);
   Ok(transport_plan.into_pyarray(py).to_owned())
}

#[pymodule]
fn rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sinkhorn))?;
    m.add_wrapped(wrap_pyfunction!(calculate_1D_ot))?;

    Ok(())
}
