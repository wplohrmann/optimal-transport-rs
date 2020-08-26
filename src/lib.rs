use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn hello()
{
    println!("Hello, world!");
}

#[pymodule]
fn optimal_transport_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(hello))?;

    Ok(())
}
