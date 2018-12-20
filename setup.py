from setuptools import setup, find_packages

setup(
    name='PQsim',
    version='0.1',
    author='Edwin Tham',
    author_email='quantum.explorer@gmail.com',
    description='Pedagogical Quantum SIMulator',
    install_requires=['numba', 'numpy'],
    packages=find_packages(),
    package_data={}
)