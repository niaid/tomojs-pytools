from pathlib import Path
import SimpleITK as sitk

from typing import Union

PathType = Union[str, Path]


def _make_image_file_reader_with_info(file_path: PathType) -> sitk.ImageFileReader:
    """
    Constructs an SimpleITK ImageFileReader and reads the header information.

    If the file does not exist SimpleITK/ITK/SWIG exception will be thrown.
    """

    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(str(file_path))
    file_reader.ReadImageInformation()
    return file_reader


def is_int16(file_path: PathType) -> bool:
    """
    Read an image file header to inspect meta-data.

    Supported file formats include TIFF, and others supported by SimpleITK and the Insight toolkit.

    :param file_path: The path to an image file.
    :returns: True if the pixel type is a signed 16-bit integer, False otherwise.
    """

    return _make_image_file_reader_with_info(file_path).GetPixelID() == sitk.sitkInt16


def is_16bit(file_path: PathType) -> bool:
    """
    Read an image file header to inspect meta-data.

    Supported file formats include TIFF, and others supported by SimpleITK and the Insight toolkit.

    :param file_path: The path to an image file.
    :returns: True if the pixel type is a 16-bit integer (signed or unsigned), False otherwise.

    """
    return _make_image_file_reader_with_info(file_path).GetPixelID() in [sitk.sitkInt16, sitk.sitkUInt16]


__all__ = ["is_int16", "is_16bit"]
