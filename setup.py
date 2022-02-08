import setuptools

setuptools.setup(
    name= 'ramCOH',
    version= '0.1',
    description= '...',
    author= 'Thomas van Gerve',
    
    packages= setuptools.find_packages(
        exclude= ['examples']
        ),

    # package_dir= {'' : 'petroPy'},
    package_data= {'ramCOH': ['static/*']}, 

    install_requires= [
    'pandas',
    'matplotlib',
    'numpy',
    'scipy',
    'csaps'
    ]
)