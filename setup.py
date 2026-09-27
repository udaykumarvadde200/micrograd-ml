import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="micrograd-ml",
    version="0.1.0",
    author="V. Uday Kumar",
    description="A from-scratch extension of micrograd with neural networks, optimizers, batching, metrics, and ML experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/udaykumarvadde200/micrograd-ml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)