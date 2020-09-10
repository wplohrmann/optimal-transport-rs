use numpy::{PyArray1, PyArray2};
use numpy::convert::IntoPyArray;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod sinkhorn;

use sinkhorn::sinkhorn as impl_sinkhorn;

#[pyfunction]
fn sinkhorn(py: Python<'_>, a: &PyArray1<f32>, b: &PyArray1<f32>, cost: &PyArray2<f32>, reg: f32) -> PyResult<Py<PyArray2<f32>>> {
   let transport_plan = impl_sinkhorn(a.to_owned_array(), b.to_owned_array(), cost.to_owned_array(), reg);
   Ok(transport_plan.into_pyarray(py).to_owned())
}

#[pymodule]
fn rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sinkhorn))?;

    Ok(())
}
