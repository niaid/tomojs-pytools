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
import numpy as np
import SimpleITK as sitk
import neuroglancer
from pathlib import Path
import jinja2
from pytools import HedwigZarrImages
from typing import Union
from pytools.utils import OMEInfo
from pytools.data import ROIRectangle, ROILabel


_rgb_shader_template = """
void main() {
  emitRGB(vec3(toNormalized(getDataValue(0)),
               toNormalized(getDataValue(1)),
               toNormalized(getDataValue(2))));
}
"""

_gray_shader_template = """
#uicontrol invlerp normalized(range=[{{range[0]}}, {{range[1]}}],
    window=[{{window[0]}}, {{window[1]}}], clamp=true)
void main() {
  emitGrayscale(normalized());
}
"""

_multichannel_template = """
#uicontrol float brightness slider(default={{brightness}}, min=-1, max=1, step=0.1)
#uicontrol float contrast slider(default={{contrast}}, min=-3, max=3, step=0.1)

{% for channel in channelArray %}
#uicontrol bool {{channel.name}} checkbox(default=true)
#uicontrol vec3 color{{channel.channel}} color(default="{{channel.color}}")
#uicontrol invlerp invlerp{{channel.channel}}(range=[{{channel.range[0]}}, {{channel.range[1]}}], \
window=[{{channel.window[0]}}, {{channel.window[1]}}], \
channel={{channel.channel}}, \
clamp=true)
{% endfor %}

void main() {
    vec3 cum = vec3(0., 0., 0.);
    {% for channel in channelArray %}
    if ({{channel.name}})
    {
        cum += color{{channel.channel}} * invlerp{{channel.channel}}(getDataValue({{channel.channel}}));
    }
    {% endfor %}
    emitRGB((cum+brightness)*exp(contrast));
}
"""


_shader_parameter_cache = {}


def _convert_si_units_from_long_to_abbr(s: str) -> str:
    """
    Convert a string with long SI units to the abbreviated form.

    :param s: The string to convert.
    :return: The string with the long SI units converted to the abbreviated form.
    """

    # A map of long SI units to their abbreviated form.
    si_units = {
        "yoctometer": "ym",
        "zeptometer": "zm",
        "attometer": "am",
        "femtometer": "fm",
        "picometer": "pm",
        "nanometer": "nm",
        "micrometer": "µm",
        "millimeter": "mm",
        "centimeter": "cm",
        "decimeter": "dm",
        "meter": "m",
        "decameter": "da",
        "hectometer": "hm",
        "kilometer": "km",
        "megameter": "Mm",
        "gigameter": "Gm",
        "terameter": "Tm",
        "petameter": "Pm",
        "exagram": "Em",
        "zettameter": "Zm",
        "yottameter": "Ym",
    }
    return si_units.get(s, s)


def _homogeneous_identity(ndim: int) -> np.array:
    """
    Create a homogeneous identity matrix of dimension ndim rows by ndim+1 columns.
    """
    return np.identity(ndim + 1)[:ndim, :]


def _sitk_offset_from_transform(tx: sitk.Transform):
    """
    Get the offset from a SimpleITK transform since the method is not exposed in the Python API.

    The offset is the translation part a MatrixOffsetBaseTransform.

    """
    # compute the offset from the transform by transforming a zero point
    return tx.TransformPoint((0,) * tx.GetDimension())


def _sitk_transform_to_ng_transform(tx: sitk.Transform, inverse: bool = True) -> neuroglancer.CoordinateSpaceTransform:
    """
    Convert a SimpleITK transform to a neuroglancer.CoordinateSpaceTransform. Assumes input transform is a 2D affine
     transform. The output is assumed to be 5D [TCZYX].
    """
    assert tx.GetDimension() == 2

    if inverse:
        tx = tx.GetInverse()
    tx_matrix = np.array(tx.GetMatrix()).reshape(tx.GetDimension(), tx.GetDimension())

    # convert [XY], to [YX]
    tx_matrix = tx_matrix[::-1, ::-1]

    M = _homogeneous_identity(5)
    # assign to the YX part of the matrix
    M[3:5, 3:5] = tx_matrix

    tx_translation = np.array(_sitk_offset_from_transform(tx))

    # convert [XY], to [YX]
    tx_translation = tx_translation[::-1]
    M[3:5, 5] = tx_translation

    output_dimensions = neuroglancer.CoordinateSpace(
        names=["t'", "c^", "z", "y", "x"], units=["", "", "µm", "µm", "µm"], scales=[1, 1, 1, 1, 1]
    )
    ng_transform = neuroglancer.CoordinateSpaceTransform(output_dimensions=output_dimensions, matrix=M)
    return ng_transform


def generate_ng_shader(path: Path, key: Union[int, str]) -> str:
    """
    Given a path to a zarr structure and a key, generate a neuroglancer shader for the image.

    This will call teh HewdigZarrImages class to get the shader parameters for the image which can be computationaly
    expensive so the results a cached.
    """
    global _shader_parameter_cache

    hwz_images = HedwigZarrImages(path)
    hwz_image = hwz_images.group(key)

    if (path, key) not in _shader_parameter_cache:
        _shader_parameter_cache[(path, key)] = hwz_image.neuroglancer_shader_parameters(middle_quantile=[0.01, 0.99])

    params = _shader_parameter_cache[(path, key)]

    if hwz_image.shader_type == "RGB":
        template = _rgb_shader_template
    elif hwz_image.shader_type == "Grayscale":
        template = _gray_shader_template
    else:
        template = _multichannel_template

    j2_template = jinja2.Template(template)
    shader_code = j2_template.render(params)

    return shader_code


def add_zarr_image(viewer_txn: neuroglancer.Viewer, zarr_path: Path, server_url: str, transform_filename=None):
    """
    With in neuroglancer viewer context, ad a zarr image to the viewer.

    :param viewer_txn: The neuroglancer viewer transaction object.
    :param zarr_path: The path to the zarr file including the key for the sub image.
    :param server_url: The url to the server hosting the zarr file. The zarr_path will be appended to this url with the
    component containing the "zarr" extensions and the subsequent key.
    :param transform_filename: The filename of a SimpleITK transform file. The provided transform file is a SimpleITK
     format. The transform maps points from the output space to the input space, and is inverted before being passed
     to neuroglancer. If None, then the identity transform is used.

    """

    zarr_root = zarr_path.parent
    zarr_key = zarr_path.name

    layername = f"{zarr_root.name}/{zarr_key}"

    if transform_filename:
        tx = sitk.ReadTransform(transform_filename)
        ng_transform = _sitk_transform_to_ng_transform(tx, inverse=True)
    else:
        # Assuming stander 5D [TCZYX] of zarr, and output in nanometer units will be the correct scale.    #
        output_dimensions = neuroglancer.CoordinateSpace(
            names=["t'", "c^", "z", "y", "x"], units=["", "", "µm", "µm", "µm"], scales=[1, 1, 1, 1, 1]
        )
        M = _homogeneous_identity(5)
        ng_transform = neuroglancer.CoordinateSpaceTransform(output_dimensions=output_dimensions, matrix=M)

    viewer_txn.layers[layername] = neuroglancer.ImageLayer(
        source=neuroglancer.LayerDataSource(f"zarr://{server_url}/{zarr_root.name}/{zarr_key}", transform=ng_transform),
        shader=generate_ng_shader(zarr_root, zarr_key),
    )


def add_roi_annotations(viewer_txn, ome_xml_filename, *, layername="roi annotation", reference_zarr=None):
    """
    Add ROI annotations to the neuroglancer viewer. The annotations are read from the OME-XML file.

    The OME-XML specifications for ROI models is here:
      https://docs.openmicroscopy.org/ome-model/5.6.3/developers/roi.html

    The ROI is specified in the image coordinate space and the image meta-data is needed to convert to the physical
     space. If the reference_zarr is provided, then the image meta-data is extracted from the zarr file. Otherwise,
     the image meta-data is extracted from the OME-XML file.

    :param viewer_txn: The neuroglancer viewer transaction object.
    :param ome_xml_filename: The path to the OME-XML file.
    :param layername: The name of the annotation layer in the viewer.
    :param reference_zarr: If specified the image meta-data is extracted from the zarr file. Otherwise, the image
     meta-data is extracted from the OME-XML file.

    """

    xml_path = Path(ome_xml_filename)

    ome_idx = 0

    with open(xml_path, "r") as fp:
        data = fp.read()
        ome_info = OMEInfo(data)

    if reference_zarr is None:
        assert ome_info.dimension_order(ome_idx) == "XYZCT"
        scales = ome_info.spacing(ome_idx)[:2]
        units = ome_info.units(ome_idx)[:2]

    else:
        # Coordinate for the ROI rectangles are in the space of an image. The dimensions/CoordinateSpace map the
        # index space to physical space and the "scales" from the reference image are needed to map the space.
        zarr_root = Path(reference_zarr).parent
        zarr_key = Path(reference_zarr).name
        hwz_images = HedwigZarrImages(zarr_root)
        hwz_image = hwz_images.group(zarr_key)

        # Convert TCZYX to XY
        scales = hwz_image.spacing[:2:-1]
        units = hwz_image.units[:2:-1]

    units = [_convert_si_units_from_long_to_abbr(u) for u in units]

    layer = neuroglancer.LocalAnnotationLayer(
        dimensions=neuroglancer.CoordinateSpace(
            names=["x", "y"],
            units=units,
            scales=scales,
        )
    )

    viewer_txn.layers[layername] = layer

    for roi_model in ome_info.roi(ome_idx):
        label = roi_model.id
        for roi in roi_model.union:
            if isinstance(roi, ROILabel):
                # Assuming that label is followed by the associated rectangle in the OME-XML file.
                label = roi.text
            elif isinstance(roi, ROIRectangle):
                layer.annotations.append(
                    neuroglancer.AxisAlignedBoundingBoxAnnotation(
                        description=label,
                        id=neuroglancer.random_token.make_random_token(),
                        point_a=roi.point_a,
                        point_b=roi.point_b,
                    )
                )
                label = "unknown"
