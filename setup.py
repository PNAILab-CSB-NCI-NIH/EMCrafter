from setuptools import setup, find_packages

setup(
    name="EMCrafter",
    version="1.0.0",
    author="Hermann Degenhardt",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PNAI-CSB-NCI-NIH/EMCrafter",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    install_requires=[
        # dependencies
    ],
)