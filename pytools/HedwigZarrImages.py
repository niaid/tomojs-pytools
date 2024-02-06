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

from pathlib import Path
import zarr
from typing import Optional, Iterable, Tuple, AnyStr, Union
from pytools.utils import OMEInfo
from pytools.HedwigZarrImage import HedwigZarrImage
import logging


logger = logging.getLogger(__name__)


class HedwigZarrImages:
    """
    Represents the set of images in a OME-NGFF ZARR structure.
    """

    def __init__(self, zarr_path: Path, read_only=True):
        """
        Initialized by the path to a root of an OME zarr structure.
        """
        # check zarr is valid
        assert zarr_path.exists()
        assert zarr_path.is_dir()
        self.zarr_store = zarr.DirectoryStore(zarr_path)
        self.zarr_root = zarr.Group(store=self.zarr_store, read_only=read_only)
        self._ome_info = None

    @property
    def ome_xml_path(self) -> Optional[Path]:
        """
        Returns the path to the OME-XML file, if it exists.
        """
        if "OME" in self.zarr_root.group_keys():
            _xml_path = Path(self.zarr_store.path) / self.zarr_root["OME"].path / "METADATA.ome.xml"
            if _xml_path.exists():
                return _xml_path

    @property
    def ome_info(self) -> Optional[AnyStr]:
        """Returns OME XML as string is if exists."""

        if self._ome_info is not None:
            return self._ome_info

        _path = self.ome_xml_path
        if not _path:
            return None
        with open(_path, "r") as fp:
            self._ome_info = OMEInfo(fp.read())
            return self._ome_info

    def get_series_keys(self) -> Iterable[str]:
        """
        Returns an iterable of strings of the names or labels of the images. Will be extracted from
        the OME-XML if available otherwise the ZARR group names.
        e.g. "label_image"
        """

        if self.ome_info:
            return self.ome_info.image_names()

        return filter(lambda x: x != "OME", self.zarr_root.group_keys())

    def group(self, name: str) -> HedwigZarrImage:
        """
        Returns a HedwigZarrImage from the given ZARR group name or path.
        """

        if self.ome_xml_path is None:
            return HedwigZarrImage(self.zarr_root[name])

        ome_index_to_zarr_group = self.zarr_root["OME"].attrs["series"]
        k_idx = ome_index_to_zarr_group.index(name)
        print(f"k_idx: {k_idx}")
        return HedwigZarrImage(self.zarr_root[name], self.ome_info, k_idx)

    def __getitem__(self, item: Union[str, int]) -> HedwigZarrImage:

        for k_idx, k in enumerate(self.get_series_keys()):
            if item == k and "OME" in self.zarr_root.group_keys():
                ome_index_to_zarr_group = self.zarr_root["OME"].attrs["series"]

                if len(ome_index_to_zarr_group) != len(set(ome_index_to_zarr_group)):
                    raise RuntimeError(f'The OME "series" contains duplicated paths: f{ome_index_to_zarr_group}')

                zarr_idx = ome_index_to_zarr_group[k_idx]
                return HedwigZarrImage(self.zarr_root[zarr_idx], self.ome_info, k_idx)

        return HedwigZarrImage(self.zarr_root[item], None, 404)

    def series(self) -> Iterable[Tuple[str, HedwigZarrImage]]:
        """An Iterable of key and HedwigZarrImages stored in the ZARR structure."""
        for k in self.get_series_keys():
            yield k, self[k]
