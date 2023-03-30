#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from setuptools import setup, find_packages

with open("README.rst", "r") as fp:
    long_description = fp.read()

with open("requirements.txt", "r") as fp:
    requirements = list(filter(bool, (line.strip() for line in fp)))

setup(
    name="tomojs_pytools",
    use_scm_version={"local_scheme": "dirty-tag"},
    author="Bradley Lowekamp",
    author_email="bioinformatics@niaid.nih.gov",
    description="Tools for generating and working with Neuroglancer precompute format.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_packages(exclude=("test",)),
    url="https://www.niaid.nih.gov/research/bioinformatics-computational-biosciences-branch",
    license="Apache 2.0",
    entry_points={
        "console_scripts": [
            "mrc2nifti = pytools.ng.mrc2nifti:main",
            "mrc_visual_min_max = pytools.ng.build_histogram:main",
            "zarr_rechunk = pytools.zarr_rechunk:main",
        ]
    },
    classifiers=[
        # The version is <1.0, and there may be API incompatibilities from minor version to minor version
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)
