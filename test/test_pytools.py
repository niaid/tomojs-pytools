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

import pytools


def test_pytools_version():
    assert pytools.__version__ != "unknown"

    from packaging.version import parse, Version

    assert isinstance(parse(pytools.__version__), Version)


def test_zarr_extract(image_ome_ngff_2d):

    img = pytools.zarr_extract_2d(image_ome_ngff_2d, 8, 8)
    assert img.GetSize()[0] <= 8
    assert img.GetSize()[1] <= 8
