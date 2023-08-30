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
import pytest
import SimpleITK as sitk
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def data_path():
    return Path(__file__).parent / "data"


@pytest.fixture(
    scope="session",
    params=[
        sitk.sitkUInt8,
        sitk.sitkInt16,
        sitk.sitkUInt16,
        sitk.sitkFloat32,
        "uint16_uniform",
        "uint8_bimodal",
        "float32_uniform",
    ],
)
def image_mrc(request, tmp_path_factory):
    if isinstance(request.param, str) and request.param == "uint16_uniform":
        print(f"Calling image_mrc with {request.param}")
        fn = f"image_mrc_{request.param.replace(' ', '_')}.mrc"

        a = np.linspace(0, 2**16 - 1, num=2**16, dtype="uint16").reshape(16, 64, 64)
        img = sitk.GetImageFromArray(a)
        img.SetSpacing([1.23, 1.23, 4.96])

    elif isinstance(request.param, str) and request.param == "uint8_bimodal":
        print(f"Calling image_mrc with {request.param}")
        fn = f"image_mrc_{request.param.replace(' ', '_')}.mrc"

        a = np.zeros([16, 16, 16], np.uint8)
        a[len(a) // 2 :] = 255
        img = sitk.GetImageFromArray(a)
        img.SetSpacing([12.3, 12.3, 56.7])
    elif isinstance(request.param, str) and request.param == "float32_uniform":
        print(f"Calling image_mrc with {request.param}")
        fn = f"image_mrc_{request.param.replace(' ', '_')}.mrc"

        a = np.linspace(0.0, 1.0, num=16 * 64 * 64, dtype=np.float32).reshape(16, 64, 64)
        print(a)
        img = sitk.GetImageFromArray(a)
        img.SetSpacing([1.23, 1.23, 4.96])
    else:
        pixel_type = request.param
        print(f"Calling image_mrc with {sitk.GetPixelIDValueAsString(pixel_type)}")
        fn = f"image_mrc_{sitk.GetPixelIDValueAsString(pixel_type).replace(' ', '_')}.mrc"

        size = [10, 9, 8]
        if pixel_type == sitk.sitkFloat32:
            a = np.linspace(0.0, 1.0, num=np.prod(size), dtype=np.float32).reshape(*size[::-1])
            img = sitk.GetImageFromArray(a)
        else:
            # image of just zeros
            img = sitk.Image([10, 9, 8], pixel_type)
        img.SetSpacing([1.1, 1.2, 1.3])

    fn = tmp_path_factory.mktemp("data").joinpath(fn)
    sitk.WriteImage(img, str(fn))
    return str(fn)


@pytest.fixture(
    scope="session",
    params=[sitk.sitkInt8, sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkUInt16, sitk.sitkFloat32],
)
def image_tiff(request, tmp_path_factory):
    pixel_type = request.param
    print(f"Calling image_tiff with {sitk.GetPixelIDValueAsString(pixel_type)}")
    fn = f"image_tiff_{sitk.GetPixelIDValueAsString(pixel_type).replace(' ', '_')}.tiff"
    img = sitk.Image([10, 9, 8], pixel_type)

    fn = tmp_path_factory.mktemp("data").joinpath(fn)
    sitk.WriteImage(img, str(fn))
    return str(fn)


@pytest.fixture(scope="session", params=[(16, 16, 16), (1, 64, 64), (1, 1024, 1024)])
def image_ome_ngff(request, tmp_path_factory):
    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image
    from ome_zarr.scale import Scaler
    import zarr

    path = tmp_path_factory.mktemp("zarr").joinpath("test_ngff_image.zarr")
    print(f"path: {path}")

    chunks = request.param if request.param is not None else (4, 128, 128)

    mean_val = 10
    size_xy = 1024
    size_z = 10
    rng = np.random.default_rng(0)
    data = rng.poisson(mean_val, size=(size_z, size_xy, size_xy)).astype(np.uint8)

    # write the image data
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)
    scaler = Scaler(max_layer=0, method="nearest")
    write_image(image=data, group=root, axes="zyx", storage_options=dict(chunks=chunks), scaler=scaler)

    return path


@pytest.fixture(
    scope="session",
    params=[
        {
            "x": 16,
            "y": 16,
        },
        {"x": 64, "y": 64, "z": 1},
        {
            "c": 1,
            "x": 4,
            "y": 64,
        },
        {"c": 1, "y": 64, "x": 4},
        {"t": 1, "c": 3, "x": 64, "z": 1, "y": 64},
    ],
)
def image_ome_ngff_2d(request, tmp_path_factory):
    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image
    import zarr

    path = tmp_path_factory.mktemp("zarr").joinpath("test_ngff_image.zarr")

    mean_val = 10
    rng = np.random.default_rng(0)
    data = rng.poisson(mean_val, size=[s for s in request.param.values()]).astype(np.uint8)

    # write the image data
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)
    write_image(image=data, group=root, axes=[a for a in request.param.keys()], scaler=None)

    return path
