from setuptools import setup, find_packages
import os

# Check if README.md exists and read its content
if os.path.exists('README.md'):
    with open('README.md', 'r') as fh:
        long_description = fh.read()
else:
    long_description = 'A CLI tool for Membership Inference Attacks using PyTorch and NumPy'

setup(
    name='Execution_Environment_MIA',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'click',
        'scikit-learn',
        'torchvision'
    ],
    entry_points={
        'console_scripts': [
            'execution_environment_mia=Execution_Environment_MIA.cli:cli'
        ],
    },
    author='Lukas Reith',
    author_email='uchut@student.kit.edu',
    description='A CLI tool for Membership Inference Attacks using PyTorch and NumPy',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://gitlab.kit.edu/kit/ps/students/bachelor-thesis-lukas-reith',  # Update with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
