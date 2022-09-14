from pathlib import Path
import SimpleITK as sitk

from typing import Union

PathType = Union[str, Path]


def file_to_uint8(in_file_path: PathType, out_file_path: PathType) -> None:
    """
    Read an image file, scale the input pixel intensity range to 0-255, and write an image as an unsigned 8-bit integer.

    Supported file formats include TIFF, and others supported by SimpleITK and the Insight toolkit.

    :param in_file_path: The file path for the source image.
    :param out_file_path: The file path to the output, the extension will be used to automatically determine the Image
      format used. The out_file_path must be different that the input_file_path.
    :returns: None

    .. note:: The conversion from to uint16 is done with floating point operations with truncation rounding. This is
       considered an implementation detail that may change.
    """

    assert in_file_path != out_file_path

    img = sitk.ReadImage(str(in_file_path))

    mm = sitk.MinimumMaximumImageFilter()
    mm.Execute(img)

    # Adjust the input's used pixel range to the full output intensity range
    output_maximum = 255
    input_minimum = mm.GetMinimum()
    input_maximum = mm.GetMaximum()

    ss = sitk.ShiftScaleImageFilter()
    ss.SetOutputPixelType(sitk.sitkUInt8)
    if input_maximum != input_minimum:
        scale = output_maximum / (input_maximum - input_minimum)
        shift = -input_minimum  # adding 0.5/scale would cause something closer to rounding, still not accurate

        ss.SetShift(shift)
        ss.SetScale(scale)
    else:
        ss.SetShift(-input_minimum)

    out = ss.Execute(img)

    sitk.WriteImage(out, str(out_file_path), useCompression=False)


__all__ = ["file_to_uint8"]
