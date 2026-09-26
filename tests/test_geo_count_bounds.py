"""公开格区域计数上下界接口 POST /api/v1/geo-count-bounds 的端到端测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ENDPOINT = "/api/v1/geo-count-bounds"


def cell(z: int, x: int, y: int, count: int) -> dict:
    return {"z": z, "x": x, "y": y, "count": count}


def bounds(lat_min, lat_max, lon_min, lon_max) -> dict:
    return {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }


WORLD = bounds(-90, 90, -180, 180)

# z=1 的四个象限格：
# NW (1,0,0) 纬[0,90) 经[-180,0)；NE (1,1,0) 纬[0,90) 经[0,180)；
# SW (1,0,1) 纬[-90,0) 经[-180,0)；SE (1,1,1) 纬[-90,0) 经[0,180)。
Z1_CELLS = [
    cell(1, 0, 0, 5),
    cell(1, 1, 0, 7),
    cell(1, 0, 1, 11),
    cell(1, 1, 1, 13),
]


def payload(cells=None, query=WORLD) -> dict:
    return {"cells": cells if cells is not None else Z1_CELLS, "bounds": query}


def relations(body: dict) -> list[tuple[int, int, int, str]]:
    return [(c["z"], c["x"], c["y"], c["relation"]) for c in body["cells"]]


# ---------------------------------------------------------------- 基本功能

def test_world_bounds_marks_everything_inside():
    response = client.post(ENDPOINT, json=payload())
    assert response.status_code == 200
    body = response.json()

    assert body["lower_bound"] == 36
    assert body["upper_bound"] == 36
    assert body["cross_antimeridian"] is False
    assert body["longitude_intervals"] == [[-180.0, 180.0]]
    assert body["total_cells"] == 4
    assert body["total_count"] == 36
    assert relations(body) == [
        (1, 0, 0, "inside"),
        (1, 1, 0, "inside"),
        (1, 0, 1, "inside"),
        (1, 1, 1, "inside"),
    ]
    for entry in body["cells"]:
        assert entry["lower_count"] == entry["count"] == entry["upper_count"]
        assert set(entry) == {
            "z", "x", "y", "count", "relation", "lower_count", "upper_count"
        }


def test_single_cell_inside_and_rest_outside():
    # 查询区闭矩形恰好等于 NE 格的半开覆盖：NE inside，其余 outside。
    query = bounds(0, 90, 0, 180)
    response = client.post(ENDPOINT, json=payload(query=query))
    assert response.status_code == 200
    body = response.json()

    assert body["lower_bound"] == 7
    assert body["upper_bound"] == 7
    by_key = {(c["z"], c["x"], c["y"]): c for c in body["cells"]}
    ne = by_key[(1, 1, 0)]
    assert ne["relation"] == "inside"
    assert (ne["lower_count"], ne["upper_count"]) == (7, 7)
    for key in ((1, 0, 0), (1, 0, 1), (1, 1, 1)):
        entry = by_key[key]
        assert entry["relation"] == "outside"
        assert (entry["lower_count"], entry["upper_count"]) == (0, 0)


def test_partial_intersection_only_counts_toward_upper_bound():
    # 中央经度带与每格都有实际交叠（每格各占一半经度），无格完全落入。
    query = bounds(-90, 90, -90, 90)
    response = client.post(ENDPOINT, json=payload(query=query))
    assert response.status_code == 200
    body = response.json()

    assert body["lower_bound"] == 0
    assert body["upper_bound"] == 36
    assert all(c["relation"] == "partial" for c in body["cells"])
    for entry in body["cells"]:
        assert (entry["lower_count"], entry["upper_count"]) == (0, entry["count"])


def test_mixed_inside_and_partial():
    # 纬[10,80]、经[-100,100]：
    # NW/NE 纬向完全在内、经向各交叠一半 -> partial；SW/SE 纬向交叠 -> partial。
    query = bounds(10, 80, -100, 100)
    response = client.post(ENDPOINT, json=payload(query=query))
    body = response.json()
    assert body["lower_bound"] == 0
    assert body["upper_bound"] == 12

    # 纬[0,90]、经[-180,90]：NW 半开覆盖[-180,0) 完全在内 -> inside(5)；
    # NE 经[0,180) 与 [-180,90] 部分交叠 -> partial(7)；南两格纬向不相交。
    query = bounds(0, 90, -180, 90)
    response = client.post(ENDPOINT, json=payload(query=query))
    body = response.json()
    assert body["lower_bound"] == 5
    assert body["upper_bound"] == 12
    by_key = {(c["z"], c["x"], c["y"]): c for c in body["cells"]}
    assert by_key[(1, 0, 0)]["relation"] == "inside"
    assert by_key[(1, 1, 0)]["relation"] == "partial"
    assert by_key[(1, 0, 1)]["relation"] == "outside"
    assert by_key[(1, 1, 1)]["relation"] == "outside"


def test_boundary_contact_is_not_intersection():
    """格边界沿用中点半开规则：仅在边界线上接触不算相交。"""
    # 查询西界 = 0：NW 格东界恰好接触查询西界，NW 必须 outside 而非 partial。
    query = bounds(0, 90, 0, 90)
    response = client.post(ENDPOINT, json=payload(query=query))
    body = response.json()
    by_key = {(c["z"], c["x"], c["y"]): c for c in body["cells"]}
    assert by_key[(1, 0, 0)]["relation"] == "outside"  # 沿 lon=0 接触
    assert by_key[(1, 0, 1)]["relation"] == "outside"  # 沿 lat=0/lon=0 接触
    assert by_key[(1, 1, 0)]["relation"] == "partial"
    assert by_key[(1, 1, 1)]["relation"] == "outside"  # 仅沿 lat=0 接触

    # 查询北界/东界贴着全球边缘，NE 格仍按 inside 处理（闭查询含边缘点）。
    query = bounds(0, 90, 0, 180)
    response = client.post(ENDPOINT, json=payload(query=query))
    body = response.json()
    assert body["lower_bound"] == 7


def test_response_cells_keep_input_order():
    shuffled = [Z1_CELLS[3], Z1_CELLS[0], Z1_CELLS[2], Z1_CELLS[1]]
    response = client.post(ENDPOINT, json=payload(cells=shuffled))
    body = response.json()
    assert [(c["z"], c["x"], c["y"]) for c in body["cells"]] == [
        (1, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)
    ]


def test_root_cell_z0():
    response = client.post(
        ENDPOINT,
        json=payload(cells=[cell(0, 0, 0, 42)], query=WORLD),
    )
    body = response.json()
    assert body["lower_bound"] == body["upper_bound"] == 42
    assert body["cells"][0]["relation"] == "inside"

    # 根格覆盖全球，任何真子区域都只能部分相交。
    response = client.post(
        ENDPOINT,
        json=payload(
            cells=[cell(0, 0, 0, 42)],
            query=bounds(-45, 45, -90, 90),
        ),
    )
    body = response.json()
    assert (body["lower_bound"], body["upper_bound"]) == (0, 42)
    assert body["cells"][0]["relation"] == "partial"


# ---------------------------------------------------------------- 跨 180° 经线

def test_antimeridian_wrap_splits_into_two_intervals():
    query = bounds(-90, 90, 170, -170)
    response = client.post(ENDPOINT, json=payload(query=query))
    assert response.status_code == 200
    body = response.json()

    assert body["cross_antimeridian"] is True
    assert body["longitude_intervals"] == [[170.0, 180.0], [-180.0, -170.0]]
    # 每个 z=1 格都与 [170,180] 或 [-180,-170] 有实际交叠，但都不完全落入。
    assert body["lower_bound"] == 0
    assert body["upper_bound"] == 36
    assert all(c["relation"] == "partial" for c in body["cells"])


def test_antimeridian_wrap_with_latitude_limit():
    query = bounds(0, 90, 170, -170)
    response = client.post(ENDPOINT, json=payload(query=query))
    body = response.json()
    assert body["lower_bound"] == 0
    assert body["upper_bound"] == 12  # 只有 NW(5)+NE(7)
    by_key = {(c["z"], c["x"], c["y"]): c for c in body["cells"]}
    assert by_key[(1, 0, 0)]["relation"] == "partial"
    assert by_key[(1, 1, 0)]["relation"] == "partial"
    assert by_key[(1, 0, 1)]["relation"] == "outside"
    assert by_key[(1, 1, 1)]["relation"] == "outside"


def test_antimeridian_wrap_deep_cell_inside():
    # z=12 格 (4095,1024)：经[179.912109375,180) 纬[0,0.0439453125)，
    # 完全落入跨 180° 查询 [170,180] ∪ [-180,-170] 的东半段。
    deep = cell(12, 4095, 1024, 9)
    query = bounds(0, 45, 170, -170)
    response = client.post(ENDPOINT, json=payload(cells=[deep], query=query))
    body = response.json()
    assert body["lower_bound"] == 9
    assert body["upper_bound"] == 9
    assert body["cells"][0]["relation"] == "inside"


def test_antimeridian_wrap_boundary_contact_not_intersection():
    # 格经[-180,-179.912..) 与 [-180,-180]（lon_max=-180 的退化段端点）：
    # 查询取 lon_min=180、lon_max=-180 时两段恰为 [180,180] 与 [-180,-180]，
    # z=1 格只在 ±180 边界点接触，必须全部 outside（接触不计相交）。
    query = bounds(-90, 90, 180, -180)
    response = client.post(ENDPOINT, json=payload(query=query))
    assert response.status_code == 200
    body = response.json()
    assert body["longitude_intervals"] == [[180.0, 180.0], [-180.0, -180.0]]
    assert body["lower_bound"] == 0
    assert body["upper_bound"] == 0
    assert all(c["relation"] == "outside" for c in body["cells"])


def test_wrap_and_non_wrap_symmetric_bounds():
    # 正向 [−170, 170] 与反向 (170, −170) 覆盖互补的经度范围：
    # 反向时与 z=1 格边界接触的情况不能误计。
    response = client.post(ENDPOINT, json=payload(query=bounds(-90, 90, -170, 170)))
    body = response.json()
    assert body["lower_bound"] == 0
    assert body["upper_bound"] == 36  # 每格都与中央带部分交叠
    assert all(c["relation"] == "partial" for c in body["cells"])


# ---------------------------------------------------------------- 混合层级

def test_mixed_levels_allowed_when_merely_edge_adjacent():
    # z=1 NE 大块与 z=2 南半球两个相邻小块（兄弟格，仅边相接）共存合法。
    cells = [
        cell(1, 1, 0, 3),
        cell(2, 0, 2, 2),
        cell(2, 1, 2, 2),
    ]
    response = client.post(ENDPOINT, json=payload(cells=cells))
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 7


def test_mixed_levels_coarse_and_fine_edge_contact_is_valid():
    # 细格 z=2 (2,2) 经[0,45) 纬[-45,0)，仅沿 lat=0 与粗格 NE 接触，无重叠。
    cells = [cell(1, 1, 0, 3), cell(2, 2, 2, 2)]
    response = client.post(ENDPOINT, json=payload(cells=cells))
    assert response.status_code == 200


def test_422_on_mixed_level_overlap():
    cells = [
        cell(1, 1, 0, 3),
        cell(2, 2, 0, 2),  # 经[0,90) 纬[45,90)，完全落在 NE 格内
    ]
    response = client.post(ENDPOINT, json=payload(cells=cells))
    assert response.status_code == 422
    assert "重叠" in response.json()["detail"]


def test_422_on_mixed_level_overlap_fine_listed_first():
    # 细格出现在粗格之前也必须检出重叠。
    cells = [cell(3, 7, 1, 4), cell(1, 1, 0, 3)]
    response = client.post(ENDPOINT, json=payload(cells=cells))
    assert response.status_code == 422
    assert "重叠" in response.json()["detail"]


def test_mixed_levels_consumed_by_real_geo_anonymize_publication():
    """与 /geo-anonymize 联动：把真实发布格（含边界）直接送入计数接口。"""
    records = [
        {"id": "obs-a", "latitude": 10.0, "longitude": 10.0},
        {"id": "obs-b", "latitude": 20.0, "longitude": 30.0},
        {"id": "obs-c", "latitude": -10.0, "longitude": -10.0},
        {"id": "obs-d", "latitude": -20.0, "longitude": -30.0},
    ]
    published = client.post(
        "/api/v1/geo-anonymize",
        json={"k": 2, "max_zoom": 3, "records": records},
    ).json()
    pub_cells = [
        {"z": c["z"], "x": c["x"], "y": c["y"], "count": c["count"]}
        for c in published["cells"]
    ]

    response = client.post(
        ENDPOINT,
        json=payload(cells=pub_cells, query=bounds(0, 90, -180, 180)),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["lower_bound"] == 2
    assert body["upper_bound"] == 2
    assert [(c["z"], c["x"], c["y"], c["relation"]) for c in body["cells"]] == [
        (3, 4, 3, "inside"),
        (3, 3, 4, "outside"),
    ]


# ---------------------------------------------------------------- 422 校验：格

def test_422_when_cells_empty():
    response = client.post(ENDPOINT, json=payload(cells=[]))
    assert response.status_code == 422
    assert "cells" in response.json()["detail"]


def test_422_when_cells_missing():
    response = client.post(ENDPOINT, json={"bounds": WORLD})
    assert response.status_code == 422


def test_422_when_cell_not_object():
    response = client.post(ENDPOINT, json=payload(cells=[[1, 0, 0, 2]]))
    assert response.status_code == 422


def test_422_when_cell_field_not_integer():
    template = cell(1, 0, 0, 2)
    for field, bad in (
        ("z", 1.0), ("z", "1"), ("z", True),
        ("x", 0.0), ("y", 0.5), ("count", 2.0), ("count", False),
    ):
        broken = dict(template)
        broken[field] = bad
        response = client.post(ENDPOINT, json=payload(cells=[broken]))
        assert response.status_code == 422, (field, bad)
        assert "整数" in response.json()["detail"]


def test_422_when_z_out_of_range():
    for bad_z in (-1, 13, 100):
        response = client.post(ENDPOINT, json=payload(cells=[cell(bad_z, 0, 0, 2)]))
        assert response.status_code == 422
        assert "z" in response.json()["detail"]


def test_422_when_xy_out_of_range():
    for bad in (
        cell(1, 2, 0, 2), cell(1, -1, 0, 2),
        cell(2, 4, 0, 2), cell(3, 0, 8, 2), cell(0, 1, 0, 2),
    ):
        response = client.post(ENDPOINT, json=payload(cells=[bad]))
        assert response.status_code == 422


def test_422_when_count_not_positive():
    for bad_count in (0, -3):
        response = client.post(
            ENDPOINT, json=payload(cells=[cell(1, 0, 0, bad_count)])
        )
        assert response.status_code == 422
        assert "count" in response.json()["detail"]


def test_422_when_cell_has_extra_or_missing_field():
    extra = cell(1, 0, 0, 2)
    extra["lat_min"] = 0.0
    assert client.post(ENDPOINT, json=payload(cells=[extra])).status_code == 422

    for field in ("z", "x", "y", "count"):
        missing = cell(1, 0, 0, 2)
        del missing[field]
        response = client.post(ENDPOINT, json=payload(cells=[missing]))
        assert response.status_code == 422


def test_422_on_duplicate_cell():
    response = client.post(
        ENDPOINT,
        json=payload(cells=[cell(2, 1, 1, 3), cell(2, 1, 1, 4)]),
    )
    assert response.status_code == 422
    assert "重复" in response.json()["detail"]


# ---------------------------------------------------------------- 422 校验：查询区

def test_422_when_bounds_missing():
    response = client.post(ENDPOINT, json={"cells": Z1_CELLS})
    assert response.status_code == 422


def test_422_when_bounds_not_object():
    response = client.post(
        ENDPOINT, json={"cells": Z1_CELLS, "bounds": [-90, 90, -180, 180]}
    )
    assert response.status_code == 422


def test_422_when_bounds_value_not_finite_number():
    for field in ("lat_min", "lat_max", "lon_min", "lon_max"):
        for bad in ("10", True, None):
            broken = dict(WORLD)
            broken[field] = bad
            response = client.post(ENDPOINT, json=payload(query=broken))
            assert response.status_code == 422, (field, bad)


def test_422_when_bounds_value_nan_or_infinity():
    # httpx 的 json= 无法序列化 NaN/Infinity，直接发送原始 JSON 文本。
    for bad_literal in ("NaN", "Infinity", "-Infinity"):
        body = (
            '{"cells": [{"z": 0, "x": 0, "y": 0, "count": 1}],'
            f' "bounds": {{"lat_min": 0.0, "lat_max": {bad_literal},'
            ' "lon_min": -180.0, "lon_max": 180.0}}'
        )
        response = client.post(
            ENDPOINT, content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert "有限数值" in response.json()["detail"]


def test_422_when_bounds_out_of_range():
    invalid = [
        bounds(-90.001, 90, -180, 180),
        bounds(-90, 90.001, -180, 180),
        bounds(-90, 90, -180.001, 180),
        bounds(-90, 90, -180, 180.001),
    ]
    for query in invalid:
        response = client.post(ENDPOINT, json=payload(query=query))
        assert response.status_code == 422
        assert "之间" in response.json()["detail"]


def test_422_when_bounds_degenerate():
    # 零高度矩形。
    response = client.post(
        ENDPOINT, json=payload(query=bounds(10, 10, -180, 180))
    )
    assert response.status_code == 422
    assert "lat_min" in response.json()["detail"]

    # 零宽度矩形（lon_min == lon_max，即使为 0）。
    response = client.post(
        ENDPOINT, json=payload(query=bounds(-90, 90, 30, 30))
    )
    assert response.status_code == 422
    assert "lon_min" in response.json()["detail"]


def test_422_when_bounds_has_extra_or_missing_field():
    extra = dict(WORLD)
    extra["padding"] = 1
    assert client.post(ENDPOINT, json=payload(query=extra)).status_code == 422

    for field in ("lat_min", "lat_max", "lon_min", "lon_max"):
        broken = dict(WORLD)
        del broken[field]
        response = client.post(ENDPOINT, json=payload(query=broken))
        assert response.status_code == 422


def test_422_on_extra_top_level_field():
    response = client.post(
        ENDPOINT, json={**payload(), "published_by": "reserve-a"}
    )
    assert response.status_code == 422
