import setuptools

setuptools.setup(
    name="piven",
    version="0.2.3",
    description="Prediction Intervals with specific value prediction",
    url="https://gitlab.com/jasperginn/piven.py",
    author="Jasper Ginn",
    author_email="jasperginn@godatadriven.com",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "pytest>=2.0.0",
        "scikit-learn>=0.23.2",
        "tensorflow>=2.3.0",
        "typer>=0.3.0",
        "pandas>=1.1.0",
    ],
    entry_points={"console_scripts": ["piven=piven.cli:app"]},
    extras_require={"dev": ["pytest", "bump2version"]},
)
