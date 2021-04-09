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

_installed_package = "tomojs_pytools"

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version(_installed_package)
except ImportError:
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        __version__ = get_distribution(_installed_package).version
    except DistributionNotFound:
        import logging

        _logger = logging.getLogger(__name__)
        _logger.warning(f'Package "{_installed_package }" is not installed. Unable to determine version!')
        __version__ = "dev"
        # package is not installed
        pass
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["__version__"]
