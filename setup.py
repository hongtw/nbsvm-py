from setuptools import setup, find_packages  

setup(  
    name = "nbsvm-py",  
    version = "0.0.2",  
    description = "nbsvm",  
    long_description = "nbsvm for python",  
    author = "HongTW",  
    author_email = "bbqlp33@gmail.com",  
    packages = find_packages(),  
    include_package_data = True,  
    platforms = "any",  
    install_requires = ['sklearn<=0.23.1', 'joblib<=0.15.1'],  
    scripts = [],  
)