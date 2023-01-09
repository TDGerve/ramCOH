import setuptools

setuptools.setup(
    name= 'ramCO2',
    version= '0.1',
    description= '...',
    author= 'Thomas van Gerve',
    
    packages= setuptools.find_packages(where=["ramCOH/"]
        ),

    # package_dir= {'' : 'petroPy'},
    package_data= {'ramCOH': ['static/*']}, 

    install_requires= [
    'pandas',
    'matplotlib',
    'numpy',
    'scipy',
    'csaps',
    ]
)