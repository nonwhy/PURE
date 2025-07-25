import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xllmx",
    version="0.0.1",
    author="nonwhy",
    description="An Open-source Toolkit for MLLM-based Real-World Image Super-Resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nonwhy/PURE",
    packages=["xllmx"],
    include_package_data=True,
)
