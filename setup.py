from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nimbusagent",
    version="0.0.1",
    author="Lee Huffman",
    author_email="lhuffman@aerisweather.com",
    description="A Basic LLM agent with basic memory, functions, and moderation support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hwleeh/nimbusagent",
    project_urls={
        "Bug Tracker": "https://github.com/hwleeh/nimbusagent/issues",
    },
    license="MIT",
    packages=find_packages(),
    install_requires=[
        'openai==0.27.8',
        'tiktoken==0.5.1',
        'pydantic==2.4.2',
        'numpy==1.24.4'
    ]
)
