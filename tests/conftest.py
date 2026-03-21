import pytest
import numpy as np

TINY_VRP_TEXT = """\
NAME : tiny
COMMENT : (Best known: 44)
TYPE : CVRP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 50
NODE_COORD_SECTION
1 0 0
2 10 0
3 10 10
4 0 10
DEMAND_SECTION
1 0
2 20
3 20
4 20
DEPOT_SECTION
1
-1
EOF
"""

@pytest.fixture
def tiny_vrp_file(tmp_path):
    p = tmp_path / "tiny.vrp"
    p.write_text(TINY_VRP_TEXT)
    return str(p)
