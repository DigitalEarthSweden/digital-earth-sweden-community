import os
import pytest
from pathlib import Path
from testbook import testbook


@pytest.mark.parametrize(
    "bookname",
    [
        "020_DATA_010-Loading-DataCubes.ipynb",
        "020_DATA_020-Property_Filtering.ipynb",
        "030_GEOM_010-Bounding-box.ipynb",
        "030_GEOM_020-Polygons-Intro.ipynb",
        "030_GEOM_030-Polygons-Areas.ipynb",
        "040_VIZ_010-RGB.ipynb",
        "040_VIZ_020-Drawing-Images-On-Base-Map.ipynb",
        "050_PROC_010-Processes-intro.ipynb",
        "050_PROC_020-Operators.ipynb",
        "060_EO_020-Reductions.ipynb",
        "060_EO_030-Spatial-Filtering.ipynb",
    ],
)
def test_test_environment_setup(bookname):
    # Check if we're in Docker by checking if /proj/tutorials exists
    if os.path.exists("/proj/tutorials"):
        notebooks = "/proj/tutorials"
    else:
        # Use the sibling directory when running locally
        notebooks = f"{Path(__file__).parent.parent}/tutorials"

    # Construct the path to the notebook
    notebook_path = f"{notebooks}/{bookname}"

    # Make sure the notebook exists before proceeding
    assert os.path.exists(notebook_path), f"Notebook {notebook_path} not found!"

    # Load the notebook using testbook
    with testbook(notebook_path, execute=False, timeout=-1) as tb:
        assert tb  # Ensures the notebook is properly loaded
