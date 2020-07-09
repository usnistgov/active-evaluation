import setuptools

setuptools.setup(
    packages=setuptools.find_packages(exclude=['tests*', 'data', 'docs', 'experiment_outputs',
                                               'experiment_scripts', 'readme_experiment_outputs']),
    install_requires=[
        'coverage',
        'gitpython',
        'joblib',
        'numpy',
        'pandas',
        'pytest',
        'scipy',
        'sklearn',
        'm2r',
        'sphinx',
        'flake8',
        'sphinx-rtd-theme'
    ]
)
