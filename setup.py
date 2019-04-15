from io import open

from setuptools import find_packages, setup

setup(
    name="dialogentail",
    version="0.1.0",
    author="Nouha Dziri, Ehsan Kamalloo",
    author_email="dziri@cs.ualberta.ca",
    description="Framework for automatic evaluation of dialogue consistency using entailment",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='dialogue evaluation semantic similarity entailment NLP deep learning',
    url="https://github.com/nouhadziri/DialogEntailment",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['allennlp>=0.8.3',
                      'pytorch-pretrained-bert',
                      'spacy>=2.1.0,2.2.0',
                      'scikit-learn',
                      'pandas',
                      'seaborn',
                      'smart_open',
                      'tqdm'],
    python_requires='>=3.6.0',
    tests_require=['pytest'],
)
