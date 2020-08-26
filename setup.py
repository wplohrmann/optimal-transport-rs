from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="optimal-transport-rs",
    version="1.0",
    rust_extensions=[RustExtension("optimal_transport_rs.optimal_transport_rs", binding=Binding.PyO3, debug=False)],
    packages=["optimal_transport_rs"],
    zip_safe=False,
)
