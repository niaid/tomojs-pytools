from pathlib import Path
import SimpleITK as sitk

from typing import Union

from pytools.utils.OMEInfo import OMEInfo

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


def read_ome_info(file_path: PathType) -> OMEInfo:
    """
    Reads the OME-XML embedded in a TIFF file's "ImageDescription" tag, without loading pixel data.

    :param file_path: The path to an OME-TIFF file.
    :returns: An OMEInfo object parsed from the file's embedded OME-XML.
    :raises ValueError: If the file has no embedded OME-XML (e.g. not an OME-TIFF).
    """

    file_reader = _make_image_file_reader_with_info(file_path)
    if "ImageDescription" not in file_reader.GetMetaDataKeys():
        raise ValueError(f"No OME-XML found in file: {file_path}")
    return OMEInfo(file_reader.GetMetaData("ImageDescription"))


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


__all__ = ["is_int16", "is_16bit", "read_ome_info"]
