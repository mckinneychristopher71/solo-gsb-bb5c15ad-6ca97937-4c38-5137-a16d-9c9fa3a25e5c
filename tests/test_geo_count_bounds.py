"""区域观测数上下界接口 POST /api/v1/geo-count-bounds 的端到端测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.geo import cell_bounds
from app.main import app

client = TestClient(app)

ENDPOINT = "/api/v1/geo-count-bounds"
GEO_ENDPOINT = "/api/v1/geo-anonymize"


def cell(z: int, x: int, y: int, count: int) -> dict:
    return {"z": z, "x": x, "y": y, "count": count}


def bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    return cell_bounds(z, x, y)


def query(cells_, lat_min, lat_max, lon_start, lon_end) -> dict:
    return {
        "cells": cells_,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_start": lon_start,
        "lon_end": lon_end,
    }


# z=2 格：经纬跨度 90°×45°。
# (x1,y1) 纬度 0..45、经度 -90..0； (x2,y1) 同纬度、经度 0..90；
# (x1,y0) 纬度 45..90、经度 -90..0；(x0,y1) 纬度 0..45、经度 -180..-90。
MIXED_CELLS = [
    cell(2, 1, 1, 3),    # 将被查询区完整包含
    cell(2, 2, 1, 5),    # 东侧仅部分相交
    cell(2, 1, 0, 7),    # 北侧仅边界接触
    cell(2, 0, 1, 11),   # 西侧仅边界接触
]


def relation_map(body: dict) -> dict[tuple[int, int, int], dict]:
    return {(c["z"], c["x"], c["y"]): c for c in body["cells"]}


# ---------------------------------------------------------------- 基本功能

def test_inside_partial_outside_classification():
    # 查询区纬度 0..45、经度 -90..20。
    response = client.post(
        ENDPOINT, json=query(MIXED_CELLS, 0, 45, -90, 20)
    )
    assert response.status_code == 200
    body = response.json()

    rel = relation_map(body)
    inside = rel[(2, 1, 1)]
    partial = rel[(2, 2, 1)]
    touch_north = rel[(2, 1, 0)]
    touch_west = rel[(2, 0, 1)]

    assert inside["relation"] == "inside"
    assert inside["lower_count"] == 3
    assert inside["upper_count"] == 3

    assert partial["relation"] == "partial"
    assert partial["lower_count"] == 0
    assert partial["upper_count"] == 5

    # 仅在纬度 45° 或经度 -90° 边界上接触的格不计入任何一界。
    assert touch_north["relation"] == "outside"
    assert touch_north["lower_count"] == touch_north["upper_count"] == 0
    assert touch_west["relation"] == "outside"
    assert touch_west["lower_count"] == touch_west["upper_count"] == 0

    # 完全落入计入两界，部分相交只计入上界。
    assert body["lower_bound"] == 3
    assert body["upper_bound"] == 3 + 5
    assert body["total_count"] == 3 + 5 + 7 + 11


def test_response_cells_keep_input_order_and_fields():
    response = client.post(
        ENDPOINT, json=query(MIXED_CELLS, 0, 45, -90, 20)
    )
    assert response.status_code == 200
    cells_ = response.json()["cells"]
    assert [(c["z"], c["x"], c["y"]) for c in cells_] == [
        (2, 1, 1), (2, 2, 1), (2, 1, 0), (2, 0, 1)
    ]
    for item in cells_:
        assert set(item) == {
            "z", "x", "y", "count", "relation",
            "lower_count", "upper_count",
        }


def test_query_box_exactly_equals_cell_is_inside():
    lat_min, lat_max, lon_min, lon_max = bounds(2, 1, 1)
    response = client.post(
        ENDPOINT,
        json=query([cell(2, 1, 1, 4)], lat_min, lat_max, lon_min, lon_max),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["cells"][0]["relation"] == "inside"
    assert body["lower_bound"] == body["upper_bound"] == 4


def test_root_cell_against_world_and_small_box():
    world = client.post(
        ENDPOINT, json=query([cell(0, 0, 0, 100)], -90, 90, -180, 180)
    )
    assert world.status_code == 200
    w = world.json()
    assert w["cells"][0]["relation"] == "inside"
    assert w["lower_bound"] == w["upper_bound"] == 100

    small = client.post(
        ENDPOINT, json=query([cell(0, 0, 0, 100)], -10, 10, -10, 10)
    )
    assert small.status_code == 200
    s = small.json()
    assert s["cells"][0]["relation"] == "partial"
    assert s["lower_bound"] == 0
    assert s["upper_bound"] == 100


def test_empty_cells_return_zero_bounds():
    response = client.post(ENDPOINT, json=query([], 0, 45, -90, 90))
    assert response.status_code == 200
    assert response.json() == {
        "lower_bound": 0,
        "upper_bound": 0,
        "total_count": 0,
        "cells": [],
    }


# ---------------------------------------------------------------- 北/东归格边界

def test_boundary_contact_on_north_edge_not_counted():
    # 查询区下沿恰好压在 (x1,y1) 格的北边界 45° 上：该格仅边接触，不计入；
    # 北侧相邻格 (x1,y0) 南边界 45° 闭合（点归北侧格），与查询区正面积交叠。
    response = client.post(
        ENDPOINT, json=query(MIXED_CELLS, 45, 80, -90, 90)
    )
    assert response.status_code == 200
    rel = relation_map(response.json())
    assert rel[(2, 1, 1)]["relation"] == "outside"
    assert rel[(2, 2, 1)]["relation"] == "outside"
    assert rel[(2, 0, 1)]["relation"] == "outside"
    north = rel[(2, 1, 0)]
    assert north["relation"] == "partial"
    assert north["upper_count"] == 7


def test_boundary_contact_on_east_edge_belongs_to_east_cell():
    # 经度 0° 是 (x1,*) 的东边界、(x2,*) 的西边界；点归东侧格。
    # 查询区从 0° 起：西格仅边接触（outside），东格完整落入（inside）。
    response = client.post(
        ENDPOINT, json=query(MIXED_CELLS, 0, 45, 0, 90)
    )
    assert response.status_code == 200
    rel = relation_map(response.json())
    assert rel[(2, 1, 1)]["relation"] == "outside"
    assert rel[(2, 2, 1)]["relation"] == "inside"
    assert response.json()["lower_bound"] == 5


def test_query_corner_touch_only_not_counted():
    # 查询区只在单点 (45, -90) 与格角接触：必须 outside，不能误计相交。
    response = client.post(
        ENDPOINT, json=query([cell(2, 1, 1, 9)], 45, 60, -120, -90)
    )
    assert response.status_code == 200
    item = response.json()["cells"][0]
    assert item["relation"] == "outside"
    assert response.json()["upper_bound"] == 0


# ---------------------------------------------------------------- 跨 180° 经线

def test_antimeridian_query_splits_into_two_segments():
    # (x3,y1) 经度 90..180；(x0,y1) 经度 -180..-90；(x1,y1) 经度 -90..0。
    cells_ = [cell(2, 3, 1, 4), cell(2, 0, 1, 6), cell(2, 1, 1, 2)]
    # 起点 170 > 终点 -170：跨 180°，两段为 [170,180] 与 [-180,-170]。
    response = client.post(
        ENDPOINT, json=query(cells_, 10, 40, 170, -170)
    )
    assert response.status_code == 200
    rel = relation_map(response.json())
    assert rel[(2, 3, 1)]["relation"] == "partial"
    assert rel[(2, 0, 1)]["relation"] == "partial"
    assert rel[(2, 1, 1)]["relation"] == "outside"
    body = response.json()
    assert body["lower_bound"] == 0
    assert body["upper_bound"] == 10


def test_antimeridian_query_can_fully_contain_edge_cells():
    # z=12 格宽约 0.0879°：最东列 x=4095 西界约 179.912、最西列 x=0 东界
    # 约 -179.912，跨 180° 查询 [170, -170] 配合全纬度可完整包含这两个格。
    east = cell(12, 4095, 2000, 4)
    west = cell(12, 0, 2000, 6)
    middle = cell(12, 2048, 2000, 8)
    response = client.post(
        ENDPOINT, json=query([east, west, middle], -90, 90, 170, -170)
    )
    assert response.status_code == 200
    rel = relation_map(response.json())
    assert rel[(12, 4095, 2000)]["relation"] == "inside"
    assert rel[(12, 0, 2000)]["relation"] == "inside"
    assert rel[(12, 2048, 2000)]["relation"] == "outside"
    body = response.json()
    assert body["lower_bound"] == body["upper_bound"] == 10


# ---------------------------------------------------------------- 混合层级

def test_mixed_zoom_disjoint_cells_accepted():
    # z=1 西北格（纬度 0..90、经度 -180..0）与东侧整块 z=2 格互不重叠。
    cells_ = [cell(1, 0, 0, 2)]
    cells_ += [
        cell(2, x, y, 1) for x in (2, 3) for y in (0, 1, 2, 3)
    ]
    response = client.post(
        ENDPOINT, json=query(cells_, -45, 45, -100, 100)
    )
    assert response.status_code == 200
    body = response.json()
    # 粗格与查询区正面积交叠但不被完整包含 -> partial；其余在东侧的 z=2
    # 格中只有经度跨过 100 的列与查询区不相交或相交，断言上下界自洽即可。
    rel = relation_map(body)
    assert rel[(1, 0, 0)]["relation"] == "partial"
    assert 0 <= body["lower_bound"] <= body["upper_bound"] <= body["total_count"]


# ---------------------------------------------------------------- 与发布接口联动

def test_consumes_geo_anonymize_cells_output():
    # k=1 时每条记录发布到 max_zoom 叶格；直接把发布格喂给上下界接口，
    # 全球查询区应把所有格计入两界，总和等于记录数。
    records = [
        {"id": "obs-a", "latitude": 31.7, "longitude": 110.6},
        {"id": "obs-b", "latitude": 10.2, "longitude": -3.3},
        {"id": "obs-c", "latitude": -40.0, "longitude": 170.0},
    ]
    pub = client.post(
        GEO_ENDPOINT, json={"k": 1, "max_zoom": 6, "records": records}
    )
    assert pub.status_code == 200
    published = pub.json()["cells"]

    fed = [
        {"z": c["z"], "x": c["x"], "y": c["y"], "count": c["count"]}
        for c in published
    ]
    response = client.post(
        ENDPOINT, json=query(fed, -90, 90, -180, 180)
    )
    assert response.status_code == 200
    body = response.json()
    assert all(c["relation"] == "inside" for c in body["cells"])
    assert body["lower_bound"] == body["upper_bound"] == body["total_count"] == 3


# ---------------------------------------------------------------- 422 校验

def test_422_when_z_out_of_range():
    for bad_z in (-1, 13, 100):
        response = client.post(
            ENDPOINT, json=query([cell(bad_z, 0, 0, 1)], 0, 1, 0, 1)
        )
        assert response.status_code == 422
        assert "z" in response.json()["detail"]


def test_422_when_xy_out_of_range():
    response = client.post(
        ENDPOINT, json=query([cell(2, 4, 0, 1)], 0, 1, 0, 1)
    )
    assert response.status_code == 422
    assert "越界" in response.json()["detail"]

    response = client.post(
        ENDPOINT, json=query([cell(2, 0, 4, 1)], 0, 1, 0, 1)
    )
    assert response.status_code == 422
    assert "越界" in response.json()["detail"]


def test_422_when_cell_indices_not_integer():
    for field, bad in (
        ("z", True), ("z", 1.5), ("z", "2"),
        ("x", True), ("x", 0.0), ("x", "0"),
        ("y", False), ("y", 2.0),
    ):
        c = cell(2, 0, 0, 1)
        c[field] = bad
        response = client.post(ENDPOINT, json=query([c], 0, 1, 0, 1))
        assert response.status_code == 422, (field, bad)


def test_422_when_count_not_positive_integer():
    for bad_count in (0, -1, True, False, 2.0, "3"):
        response = client.post(
            ENDPOINT, json=query([cell(1, 0, 0, bad_count)], 0, 1, 0, 1)
        )
        assert response.status_code == 422, bad_count


def test_422_when_cell_missing_field():
    response = client.post(
        ENDPOINT,
        json={"cells": [{"z": 1, "x": 0, "y": 0}],
              "lat_min": 0, "lat_max": 1, "lon_start": 0, "lon_end": 1},
    )
    assert response.status_code == 422


def test_422_on_duplicate_cell():
    response = client.post(
        ENDPOINT,
        json=query([cell(1, 0, 0, 2), cell(1, 0, 0, 3)], 0, 10, 0, 10),
    )
    assert response.status_code == 422
    assert "重复" in response.json()["detail"]


def test_422_on_mixed_zoom_overlap():
    # z=1 (x0,y0) 覆盖纬度 0..90、经度 -180..0；z=2 (x1,y1)
    # （纬度 0..45、经度 -90..0）是它的子格，地理区域重叠。
    response = client.post(
        ENDPOINT,
        json=query([cell(1, 0, 0, 2), cell(2, 1, 1, 4)], 0, 10, 0, 10),
    )
    assert response.status_code == 422
    assert "重叠" in response.json()["detail"]


def test_mixed_zoom_cells_sharing_only_edge_are_allowed():
    # z=1 西北格与 z=2 西南子格 (x0,y2，纬度 -45..0) 仅在纬度 0° 共边。
    response = client.post(
        ENDPOINT,
        json=query([cell(1, 0, 0, 2), cell(2, 0, 2, 4)], -90, 90, -180, 180),
    )
    assert response.status_code == 200


def test_422_when_lat_bounds_invalid():
    good_cells = [cell(1, 0, 0, 1)]
    for bad in (
        query(good_cells, -91, 10, 0, 10),
        query(good_cells, 10, 91, 0, 10),
        query(good_cells, 45, 45, 0, 10),
        query(good_cells, 80, 10, 0, 10),
    ):
        response = client.post(ENDPOINT, json=bad)
        assert response.status_code == 422


def test_422_when_lon_bounds_invalid():
    good_cells = [cell(1, 0, 0, 1)]
    for bad in (
        query(good_cells, 0, 10, -200, 10),
        query(good_cells, 0, 10, 0, 200),
        query(good_cells, 0, 10, 50, 50),
    ):
        response = client.post(ENDPOINT, json=bad)
        assert response.status_code == 422


def test_422_when_bounds_not_finite_number():
    # 布尔、字符串、None 等非数值边界必须 422。
    for field, bad in (
        ("lat_min", True), ("lat_min", "0"), ("lat_min", None),
        ("lat_max", False), ("lat_max", 12.5),
        ("lon_start", True), ("lon_start", []),
        ("lon_end", "10"), ("lon_end", {"v": 1}),
    ):
        body = query([cell(1, 0, 0, 1)], 0, 10, 0, 10)
        body[field] = bad
        response = client.post(ENDPOINT, json=body)
        # 12.5 替换 lat_max 后仍满足 0 < 12.5，属于合法数值 -> 200；
        # 其余都不是合法数值 -> 422。
        if field == "lat_max" and bad == 12.5:
            assert response.status_code == 200, (field, bad)
        else:
            assert response.status_code == 422, (field, bad)

    # 整数边界合法（JSON 数字不区分整/浮点）。
    ok = client.post(
        ENDPOINT, json=query([cell(1, 0, 0, 1)], 0, 10, 0, 10)
    )
    assert ok.status_code == 200

    # NaN / Infinity 经原始 JSON 文本发送。
    for literal in ("NaN", "Infinity", "-Infinity"):
        raw = (
            '{"cells":[{"z":1,"x":0,"y":0,"count":1}],'
            f'"lat_min":0,"lat_max":10,"lon_start":0,"lon_end":{literal}}}'
        )
        response = client.post(
            ENDPOINT, content=raw, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert "有限数值" in response.json()["detail"]


def test_422_on_missing_or_extra_body_fields():
    good_cells = [cell(1, 0, 0, 1)]

    missing = client.post(
        ENDPOINT,
        json={"cells": good_cells, "lat_min": 0, "lat_max": 10, "lon_start": 0},
    )
    assert missing.status_code == 422

    extra = client.post(
        ENDPOINT,
        json={**query(good_cells, 0, 10, 0, 10), "padding": 1},
    )
    assert extra.status_code == 422

    extra_cell = client.post(
        ENDPOINT,
        json=query([{**cell(1, 0, 0, 1), "k": 5}], 0, 10, 0, 10),
    )
    assert extra_cell.status_code == 422


def test_422_when_cell_is_not_object_or_cells_not_list():
    response = client.post(
        ENDPOINT, json=query([[1, 0, 0, 1]], 0, 10, 0, 10)
    )
    assert response.status_code == 422

    response = client.post(
        ENDPOINT,
        json={"cells": {"z": 1}, "lat_min": 0, "lat_max": 10,
              "lon_start": 0, "lon_end": 10},
    )
    assert response.status_code == 422


def test_root_lists_new_endpoint():
    body = client.get("/").json()
    assert body["geo_count_bounds"] == "/api/v1/geo-count-bounds"
