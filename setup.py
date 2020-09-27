from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="optimal-transport",
    version="0.1",
    rust_extensions=[RustExtension("optimal_transport.rust", binding=Binding.PyO3, debug=True)],
    packages=["optimal_transport"],
    install_requires=["numpy", "matplotlib"],
    zip_safe=False,
)
