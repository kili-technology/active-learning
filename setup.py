from setuptools import find_packages, setup

import al

setup(
    name="al",
    version=al.__version__,
    packages=find_packages(),
    install_requires=[
        "torch>=1.3",
        "torchvision>=0.3",
        "opencv-python~=4.0",
        "yacs==0.1.6",
        "Vizer~=0.1.4",
    ],
    author="Maxime Duval",
    author_email="maxime@kili-technology.com",
    description="Experiments around active learning",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mDuval1/active-learning",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    python_requires=">=3.6",
    include_package_data=True,
)
