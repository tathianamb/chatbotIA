from setuptools import setup, find_packages

setup(
    name="myvecstore",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "faiss-cpu",
        "nomic-embed",
    ],
    author="Tathiana Barchi",
    description="Python package for creating Vector Store using FAISS",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/seu_usuario/meu_pacote",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Versão mínima do Python
)
