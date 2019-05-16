from setuptools import setup, find_packages  

setup(  
    name = "nbsvm-py",  
    version = "0.1",  
    description = "nbsvm",  
    long_description = "nbsvm for python",  
    author = "Hong",  
    author_email = "bbqlp33@gmail.com",  
    packages = find_packages(),  
    include_package_data = True,  
    platforms = "any",  
    install_requires = ['sklearn'],  
    scripts = [],  
)