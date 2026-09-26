"""区域观测数上下界：对已发布的公开格做经纬矩形查询的计数归属。

调用方提交一组 ``/geo-anonymize`` 发布过的 ``z/x/y`` 格及每格计数，再给一个
经纬矩形查询区。仅依据公开格无法得到区域内观测数的准确值（格内观测点位置
未知），只能给出：

- **下界**：完全落入查询区（含边界）的格计数之和——这些格的观测必在区内；
- **上界**：与查询区有正面积交叠的格计数之和——完全落入与部分相交的格都
  可能向区内贡献观测；仅边界接触的格不计入；
- 与查询区完全不相交的格不计入任何一界。

格几何沿用 geo 模块：以 -180～180、-90～90 为根格按中点递归四分，y 向北
为 0。归属判定与 geo 模块“边界点归入最北、最东子格”的规则保持一致，并以
**正面积交叠**为准：完全落入（查询区闭矩形完整覆盖格，边界可重合）计入
两界，仅有正面积交叠计入上界，仅在边或点上接触一律视为不相交，避免边界
接触被误计为相交。查询经度起点大于终点时按跨 180° 经线处理：拆成
``[起点, 180]`` 与 ``[-180, 终点]`` 两个区间。
"""

from __future__ import annotations

from typing import Any

from .anonymizer import unprocessable
from .geo import MAX_ZOOM_LIMIT, cell_bounds

# 格与查询区的三种归属：完全落入（计入两界）、部分相交（仅计入上界）、
# 不相交（均不计入）。
RELATION_INSIDE = "inside"
RELATION_PARTIAL = "partial"
RELATION_OUTSIDE = "outside"


def prepare_cells(
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """校验发布格列表，返回按输入顺序补充了经纬边界的格字典。

    每格必须含整数 ``z/x/y``（z 在 0～MAX_ZOOM_LIMIT，x/y 在该层合法范围）
    与正整数 ``count``；同一格编号不得重复；任意两格（含混合层级）的地理
    区域不得重叠，否则发布计数无法简单相加。
    """
    prepared: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for index, cell in enumerate(cells):
        label = f"第 {index + 1} 个格"

        z = cell.get("z")
        x = cell.get("x")
        y = cell.get("y")
        if not isinstance(z, int) or isinstance(z, bool):
            raise unprocessable(f"{label}的 z 必须是整数")
        if not isinstance(x, int) or isinstance(x, bool):
            raise unprocessable(f"{label}的 x 必须是整数")
        if not isinstance(y, int) or isinstance(y, bool):
            raise unprocessable(f"{label}的 y 必须是整数")
        if not 0 <= z <= MAX_ZOOM_LIMIT:
            raise unprocessable(f"{label}的 z 必须是 0 到 {MAX_ZOOM_LIMIT} 之间的整数")

        n = 1 << z
        if not 0 <= x < n or not 0 <= y < n:
            raise unprocessable(
                f"{label}的格编号 (z={z}, x={x}, y={y}) 越界："
                f"z={z} 层 x/y 必须在 0 到 {n - 1} 之间"
            )
        if (z, x, y) in seen:
            raise unprocessable(
                f"{label}的格编号 (z={z}, x={x}, y={y}) 与前面的格重复："
                "同一发布格只能出现一次"
            )
        seen.add((z, x, y))

        count = cell.get("count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise unprocessable(f"{label}的 count 必须是正整数")
        if count < 1:
            raise unprocessable(f"{label}的 count 必须是正整数（>= 1）")

        lat_min, lat_max, lon_min, lon_max = cell_bounds(z, x, y)
        prepared.append(
            {
                "z": z,
                "x": x,
                "y": y,
                "count": count,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            }
        )

    for i in range(len(prepared)):
        for j in range(i):
            if _cells_overlap(prepared[j], prepared[i]):
                a, b = prepared[j], prepared[i]
                raise unprocessable(
                    f"第 {j + 1} 个格 (z={a['z']}, x={a['x']}, y={a['y']}) 与"
                    f"第 {i + 1} 个格 (z={b['z']}, x={b['x']}, y={b['y']}) "
                    "地理区域重叠：混合层级的发布格必须互不重叠"
                )

    return prepared


def _cells_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """判断两个四分格的地理区域是否重叠（含混合层级）。

    同一四叉树的格要么完全不相交，要么其中一个整体包含另一个（含完全相同）。
    同层级仅在 x/y 都相同时重叠；不同层级时较粗格包含较细格，当且仅当细格
    编号整除粗格跨度后落在粗格编号上。仅边界接触不算重叠（不同格的边界
    重合是网格正常拼接），这里的包含关系不产生误判。
    """
    if a["z"] == b["z"]:
        return a["x"] == b["x"] and a["y"] == b["y"]
    coarse, fine = (a, b) if a["z"] < b["z"] else (b, a)
    shift = fine["z"] - coarse["z"]
    return fine["x"] >> shift == coarse["x"] and fine["y"] >> shift == coarse["y"]


def count_bounds(
    cells: list[dict[str, Any]],
    lat_min: float,
    lat_max: float,
    lon_start: float,
    lon_end: float,
) -> dict[str, Any]:
    """计算经纬矩形查询区内区域观测数的下界与上界及各格归属。

    ``lon_start > lon_end`` 表示查询区跨 180° 经线，按 ``[lon_start, 180]``
    与 ``[-180, lon_end]`` 两个经度区间处理。返回各格（与输入同序）的
    归属与上下界贡献，以及汇总的 ``lower_bound``/``upper_bound``。
    """
    # 经度区间列表：普通查询单区间；跨 180° 经线时拆成首尾两段。
    lon_segments = (
        [(lon_start, 180.0), (-180.0, lon_end)]
        if lon_start > lon_end
        else [(lon_start, lon_end)]
    )

    lower_bound = 0
    upper_bound = 0
    results: list[dict[str, Any]] = []

    for cell in cells:
        relation = _classify(cell, lat_min, lat_max, lon_segments)
        # 完全落入计入两界；部分相交只计入上界；不相交不计入。
        in_lower = relation == RELATION_INSIDE
        in_upper = relation in (RELATION_INSIDE, RELATION_PARTIAL)
        lower_count = cell["count"] if in_lower else 0
        upper_count = cell["count"] if in_upper else 0
        lower_bound += lower_count
        upper_bound += upper_count
        results.append(
            {
                "z": cell["z"],
                "x": cell["x"],
                "y": cell["y"],
                "count": cell["count"],
                "relation": relation,
                "lower_count": lower_count,
                "upper_count": upper_count,
            }
        )

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "total_count": sum(cell["count"] for cell in cells),
        "cells": results,
    }


def _classify(
    cell: dict[str, Any],
    lat_min: float,
    lat_max: float,
    lon_segments: list[tuple[float, float]],
) -> str:
    """判定单个格相对查询区（纬度范围 + 经度区间列表）的归属。"""
    if _lat_contains(cell, lat_min, lat_max) and any(
        _lon_contains(cell, seg_start, seg_end) for seg_start, seg_end in lon_segments
    ):
        return RELATION_INSIDE

    if _lat_positive_overlap(cell, lat_min, lat_max) and any(
        _lon_positive_overlap(cell, seg_start, seg_end)
        for seg_start, seg_end in lon_segments
    ):
        return RELATION_PARTIAL

    return RELATION_OUTSIDE


def _lat_contains(cell: dict[str, Any], lat_min: float, lat_max: float) -> bool:
    """查询纬度闭区间是否完整覆盖格的纬度范围。"""
    return cell["lat_min"] >= lat_min and cell["lat_max"] <= lat_max


def _lon_contains(cell: dict[str, Any], lon_start: float, lon_end: float) -> bool:
    """单个经度闭区间是否完整覆盖格的经度范围。"""
    return cell["lon_min"] >= lon_start and cell["lon_max"] <= lon_end


def _lat_positive_overlap(
    cell: dict[str, Any], lat_min: float, lat_max: float
) -> bool:
    """格与查询纬度范围是否存在正长度交叠；仅端点接触不算。"""
    lo = max(cell["lat_min"], lat_min)
    hi = min(cell["lat_max"], lat_max)
    return lo < hi


def _lon_positive_overlap(
    cell: dict[str, Any], lon_start: float, lon_end: float
) -> bool:
    """格与单个经度区间是否存在正长度交叠；仅端点接触不算。"""
    lo = max(cell["lon_min"], lon_start)
    hi = min(cell["lon_max"], lon_end)
    return lo < hi
