import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="ramCOH",
    version="1.0",
    description="Library for processing and peak fitting of Raman spectra, targeted at CO2 fluids and hydrous silicate glasses",
    author="Thomas van Gerve",
    url="https://github.com/TDGerve/ramCOH",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where=["ramCOH/"]),
    package_data={"ramCOH": ["static/*"]},
    install_requires=[
        "pandas",
        "matplotlib",
        "numpy",
        "scipy",
        "csaps",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
