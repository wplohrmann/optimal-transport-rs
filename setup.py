from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="optimal-transport",
    version="1.0",
    rust_extensions=[RustExtension("optimal_transport.rust", binding=Binding.PyO3, debug=False)],
    packages=["optimal_transport"],
    zip_safe=False,
)
