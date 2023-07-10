import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scMMT",
    version="1.0.0",
    author="Songqi Zhou",
    author_email="zhousongqi21@gmail.com",
    description="A package for cell annotation, protein prediction, and low dimensional embedding representation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SongqiZhou/scMMT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=['torch>=2.0.0',  'scanpy>=1.9.3', 'scikit-learn>=1.2.2','scikit-learn-intelex>=2023.1.1','pandas>=2.0.1', 'numpy>=1.24.3', 'scipy>=1.10.1', 'tqdm>=4.65.0', 'anndata>=0.9.1'],
    python_requires=">=3.10",
)