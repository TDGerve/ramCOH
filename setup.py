import setuptools

setuptools.setup(
    name= 'ramCOH',
    version= '1.0',
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
