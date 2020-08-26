from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="optimal-transport",
    version="0.1",
    rust_extensions=[RustExtension("optimal_transport.rust", binding=Binding.PyO3, debug=False)],
    packages=["optimal_transport"],
    zip_safe=False,
)
