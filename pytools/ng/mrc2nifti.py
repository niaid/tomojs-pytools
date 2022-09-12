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
import SimpleITK as sitk
import click
import logging
import itertools
from functools import wraps
from functools import reduce
from pytools import __version__


# This function decorator is based on the SimpleITK SliceBySlice and comes for the sitk-ibex project


def sub_volume_execute(inplace=True):
    """
    A function decorator which executes func on each sub-volume and *in-place* pastes the output input the
    input image.

    :param inplace:
    :param func: A function which take a SimpleITK Image as it's first argument and returns the results.
    :return: A wrapped function.
    """

    def wrapper(func):
        @wraps(func)
        def slice_by_slice(image: sitk.Image, *args, **kwargs):

            dim = image.GetDimension()
            iter_dim = 2

            if dim <= iter_dim:
                image = func(image, *args, **kwargs)
                return image

            extract_size = list(image.GetSize())
            extract_size[iter_dim:] = itertools.repeat(0, dim - iter_dim)

            extract_index = [0] * dim
            paste_idx = [slice(None, None)] * dim

            extractor = sitk.ExtractImageFilter()
            extractor.SetSize(extract_size)
            if inplace:
                for high_idx in itertools.product(*[range(s) for s in image.GetSize()[iter_dim:]]):
                    extract_index[iter_dim:] = high_idx
                    extractor.SetIndex(extract_index)

                    paste_idx[iter_dim:] = high_idx
                    image[paste_idx] = func(extractor.Execute(image), *args, **kwargs)

            else:
                img_list = []
                for high_idx in itertools.product(*[range(s) for s in image.GetSize()[iter_dim:]]):
                    extract_index[iter_dim:] = high_idx
                    extractor.SetIndex(extract_index)

                    paste_idx[iter_dim:] = high_idx

                    img_list.append(func(extractor.Execute(image), *args, **kwargs))

                for d in range(iter_dim, dim):
                    step = reduce((lambda x, y: x * y), image.GetSize()[d + 1 :], 1)

                    join_series_filter = sitk.JoinSeriesImageFilter()
                    join_series_filter.SetSpacing(image.GetSpacing()[d])
                    join_series_filter.SetOrigin(image.GetOrigin()[d])

                    img_list = [join_series_filter.Execute(img_list[i::step]) for i in range(step)]

                assert len(img_list) == 1
                image = img_list[0]

            return image

        return slice_by_slice

    return wrapper


@sub_volume_execute(inplace=False)
def _img_convert_type(img: sitk.Image, output_type) -> sitk.Image:
    """
    Convert the img into the desired pixel type. The method safely ( without overflow/underflow ) converts from one
     pixel range to another.

    :param img: a SimpleITK Image object
    :param output_type: a SimpleITK PixelID such as sitkUInt8, sitkFloat32 for the pixel type of the returned image
    """

    # the sub_volume_execute

    if img.GetPixelID() == sitk.sitkInt8 and output_type == sitk.sitkUInt8:
        img = sitk.Cast(img, sitk.sitkInt16)
        img += 128
        return sitk.Cast(img, output_type)
    elif img.GetPixelID() == sitk.sitkInt16 and output_type == sitk.sitkUInt8:
        img = sitk.Cast(img, sitk.sitkInt32)
        img += 32768
        img /= 256
        return sitk.Cast(img, output_type)
    elif img.GetPixelID() == sitk.sitkInt16 and output_type == sitk.sitkUInt16:
        img = sitk.Cast(img, sitk.sitkInt32)
        img += 32768
        return sitk.Cast(img, output_type)
    else:
        raise Exception(
            f"Converting from {img.GetPixelIDTypeAsString()} to "
            f"{sitk.GetPixelIDValueAsString(output_type)} is not implemented."
        )


@click.command()
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument("output_image", type=click.Path(exists=False, dir_okay=False, resolve_path=True))
@click.version_option(__version__)
def main(input_image, output_image):
    """Reads the INPUT_IMAGE as an MRC formatted file to OUTPUT_IMAGE as a NIFTI formatted file.

    The OUTPUT_IMAGE file name must not already exist.
    """

    logger = logging.getLogger()
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)
    reader = sitk.ImageFileReader()
    reader.SetFileName(input_image)
    reader.SetImageIO("MRCImageIO")

    logger.info(f'Reading "{reader.GetFileName()}" image information...')

    reader.ReadImageInformation()

    logger.info(f"\tPixel Type: {sitk.GetPixelIDValueAsString(reader.GetPixelIDValue())}")
    logger.info(f"\tSize: {reader.GetSize()}")
    logger.info(f"\tSpacing: {reader.GetSpacing()}")
    logger.info(f"\tOrigin:  {reader.GetOrigin()}")

    logger.info(f'Reading "{reader.GetFileName()}" image data...')
    img = reader.Execute()

    del reader

    # Pixel types which need conversion to format supported by Neuroglancer precompute
    output_pixel_id_map = {
        sitk.sitkInt8: sitk.sitkUInt8,
        sitk.sitkInt16: sitk.sitkUInt16,
    }

    if img.GetPixelID() in output_pixel_id_map:
        output_pixel_id = output_pixel_id_map[img.GetPixelID()]
        logger.info(
            f"Converting image from {img.GetPixelIDTypeAsString()} to "
            f"{sitk.GetPixelIDValueAsString(output_pixel_id)}."
        )
        img = _img_convert_type(img, output_pixel_id)

    #  angstrom to mm
    spacing_factor_to_mm = 1e-7

    logger.info(f"Scaling spacing by {spacing_factor_to_mm} to convert to millimeters.")

    # Convert MRC spacing to NIFTI millimeters
    img.SetSpacing([s * spacing_factor_to_mm for s in img.GetSpacing()])
    img.SetOrigin([s * spacing_factor_to_mm for s in img.GetOrigin()])

    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_image)
    writer.SetImageIO("NiftiImageIO")

    logger.info(f'Writing "{writer.GetFileName()}"...')
    logger.info(f"\tPixel Type: {sitk.GetPixelIDValueAsString(img.GetPixelIDValue())}")
    logger.info(f"\tSize: {img.GetSize()}")
    logger.info(f"\tSpacing: {img.GetSpacing()}")
    logger.info(f"\tOrigin:  {img.GetOrigin()}")
    writer.Execute(img)


if __name__ == "__main__":
    main()
