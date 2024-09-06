import sys
from pathlib import Path

import pytest
from testbook import testbook

notebooks = f"{Path(__file__).parent.parent}/tutorials/"


@pytest.fixture(autouse=True)
def add_module_dir_to_sys_path():
    # Add the directory containing your modules to the Python path
    module_dir = notebooks
    print("Adding notebooks to sys path", module_dir)
    sys.path.insert(0, module_dir)
    yield
    # Remove the directory from the Python path when the test is finished
    sys.path.remove(module_dir)


# ------------------------------------------------------------------------
#                          test_test_environment_setup
# ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bookname",
    [
        # "000-Terms-and-Conditions.ipynb",
        # "010_INTRO-010-System-Overview.ipynb",
        # "010_INTRO-020-User-Handling.ipynb",
        "020_DATA_010-Loading-DataCubes.ipynb",
        "020_DATA_020-Property_Filtering.ipynb",
        "030_GEOM_010-Bounding-box.ipynb",
        "030_GEOM_020-Polygons-Intro.ipynb",
        "030_GEOM_030-Polygons-Areas.ipynb",
        "040_VIZ_010-RGB.ipynb",
        "040_VIZ_020-Drawing-Images-On-Base-Map.ipynb",
        # "040_VIZ_030-Difference-Analysis.ipynb",
        "050_PROC_010-Processes-intro.ipynb",
        "050_PROC_020-Operators.ipynb",
        # "060_EO_010-Masking-data.ipynb",
        "060_EO_020-Reductions.ipynb",
        "060_EO_020-Spatial-Filtering.ipynb",
        # "0990_Further-Reading.ipynb",
    ],
)
def test_test_environment_setup(bookname):
    with testbook(f"{notebooks}/{bookname}", execute=False, timeout=-1) as tb:
        tb.inject("import sys")
        tb.inject(f"sys.path.insert(0, '{notebooks}')")
        try:
            for i in range(0, len(tb.cells)):
                res = tb.execute_cell(i)
                print("The result was", res)
        except Exception as e:
            if "Bad Gateway" in str(e):
                raise Exception(
                    "Sporadic communication problems, just re-run the test suite!"
                )
            else:
                raise e
