import setuptools

setuptools.setup(
    name= 'ramCOH',
    version= '1.0',

    description= 'Library for processing and peak fitting of Raman spectra, with specific applications for calculating CO2 vapour density from Diad splitting.',

    author= 'Thomas van Gerve',
    
    packages= setuptools.find_packages(where=["src/"]
        ),

    package_data= {'ramCOH': ['static/*']}, 

    install_requires= [
    'pandas',
    'matplotlib',
    'numpy',
    'scipy',
    'csaps',
    ]
)
