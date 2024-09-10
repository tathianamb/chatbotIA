from setuptools import setup, find_packages

setup(
    name="MyVecStore",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-community",
        "mysql-connector-python",
        "jq"
    ],
    python_requires=">=3.12",
    author="IBICT/Tathiana Barchi",
    description="Python package for creating Vector Store using FAISS",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tathianamb/MyVecStore",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)
