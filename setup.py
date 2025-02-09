from setuptools import setup, find_packages

setup(
    name="GenerativeNetworkModels",
    version="0.1",
    description="""
    This package provides computationally efficient tools for implementing Weighted 
    Generative Models (WGMs) in network neuroscience. Unlike Binary Generative Models (BGMs), 
    WGMs capture the strength of connections between network nodes. Optimized in Python, 
    these tools offer an intuitive, graph-theoretic approach to modeling connectomes, 
    improving efficiency over existing implementations such as the Brain Connectivity 
    Toolbox and recent WGM research code.""",
    url="https://github.com/EdwardJamesYoung/GenerativeNetworkModels",
    author="Edward Young, Francesco Poli, William Mills",
    author_email="ey245@cam.ac.uk, francesco.poli@mrc-cbu.cam.ac.uk, william.mills@mrc-cbu.cam.ac.uk",
    license="",
    packages=find_packages(),
    install_requires=[
        "jaxtyping==0.2.36",
        "setuptools==75.1.0",
        "six==1.16.0",
        "sympy==1.13.3",
        "tqdm==4.66.5",
        "typeguard==2.13.3",
        "typing-extensions==4.12.2",
        "urllib3==2.3.0",
        "wheel==0.44.0",
        "zipp==3.21.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Framework :: Jupyter",
        # need to add license
    ],
)
