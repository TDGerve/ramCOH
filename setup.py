import setuptools

setuptools.setup(
    name="ramCOH",
    version="1.0",
    description="Library for processing and peak fitting of Raman spectra, targeted at CO2 fluids and hydrous silicate glasses",
    author="Thomas van Gerve",
    url="https://github.com/TDGerve/ramCOH",
    packages=setuptools.find_packages(where=["ramCOH/"]),
    package_data={"ramCOH": ["static/*"]},
    install_requires=[
        "pandas",
        "matplotlib",
        "numpy",
        "scipy",
        "csaps",
    ],
    python_requires=">=3.8",
)
