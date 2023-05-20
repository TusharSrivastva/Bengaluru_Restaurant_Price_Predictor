from setuptools import find_packages, setup
from typing import List

HYPHEN_E = '-e .'
def get_requirements(file: str)->List[str]:
    '''
    This function reads from requirements file and return a list of 
    requirements
    '''

    with open(file, mode='r') as f:
        requirements = f.readlines()
    
    requirements = [i.replace('\n','') for i in requirements]
    if HYPHEN_E in requirements:
        requirements.remove(HYPHEN_E)

    return requirements


setup(
    name = 'Bengaluru Restaurant Price Predictor',
    version = '0.0.1',
    description = 'Given some input data(rating, online delivery, booking, etc) this package predicts restraunt cost for two people in Bengaluru',
    author = 'Tushar Srivastava',
    author_email = 'tusharsrivastva1@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    )