from setuptools import setup

setup(
    name="src",
    version="0.1",
    description="An empirical study on robustness evaluation of counterfactual explanations produced by "
    "generative models for image data",
    author="Kseniya Sahatova",
    author_email="kseniya.sahatova@kuleuven.be",
    packages=[
        "src",
        "src.datasets",
        "src.models",
        "src.cf_methods",
        "src.evaluation",
        "src.utils",
    ],
    include_package_data=True,
)
