"""公开格区域计数的上下界：仅依据 /geo-anonymize 发布的 z/x/y 格估计观测数。

给定一组可发布格（每格给出 z/x/y 与格内观测数 count）和一个经纬矩形
查询区，把每格按其实际覆盖区域（边界沿用 geo 模块的中点规则）分为三类：

- ``inside``（完全落入）：整个格子都在查询区内，格内观测必然全部在区内，
  计数同时计入下界与上界；
- ``partial``（部分相交）：格子与查询区有实际交叠但未完全落入，格内观测
  可能只有部分在区内，计数只计入上界；
- ``outside``（不相交）：格子与查询区无实际交叠，计数不计入任何一界；
  仅在格边界线上接触不算相交（见 :func:`tile_bounds` 的归格说明）。

查询经度起点大于终点时，查询区按跨 180° 经线处理为两个经度区间
``[lon_min, 180]`` 与 ``[-180, lon_max]`` 的并集。

发布格允许来自不同层级（自适应四分的发布格本就如此），但同一地理位置
不允许被两个格以正面积重复覆盖，否则 count 无法直接相加，返回 422；
格之间仅边界相接是合法的（半开区间下边界只归其中一侧）。
"""

from __future__ import annotations

import math
from typing import Any

from .anonymizer import unprocessable
from .geo import MAX_ZOOM_LIMIT, ROOT_LAT_MAX, ROOT_LAT_MIN, ROOT_LON_MAX, ROOT_LON_MIN

# 三态归属标签。
RELATION_INSIDE = "inside"
RELATION_PARTIAL = "partial"
RELATION_OUTSIDE = "outside"


def prepare_cells(raw_cells: Any) -> list[tuple[int, int, int, int]]:
    """校验请求中的格列表，返回 (z, x, y, count) 列表（保持输入顺序）。

    每格必须只含整数 z/x/y 与正整数 count（拒绝布尔与非整数）：
    ``0 <= z <= MAX_ZOOM_LIMIT``，``0 <= x, y < 2^z``，``count >= 1``，
    且不允许出现未声明字段。格列表不能为空。同一 z/x/y 重复、不同层级
    格之间存在正面积重叠同样返回 422。
    """
    if not isinstance(raw_cells, list) or not raw_cells:
        raise unprocessable("cells 必须是至少包含 1 个格的非空数组")

    cells: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for index, raw in enumerate(raw_cells):
        label = f"第 {index + 1} 个格"
        if not isinstance(raw, dict):
            raise unprocessable(f"{label}必须是包含 z/x/y 与 count 的对象")

        if set(raw) != {"z", "x", "y", "count"}:
            raise unprocessable(f"{label}只能包含 z、x、y、count 四个字段")

        z = _strict_int(raw.get("z"), "z", label)
        x = _strict_int(raw.get("x"), "x", label)
        y = _strict_int(raw.get("y"), "y", label)
        count = _strict_int(raw.get("count"), "count", label)

        if not 0 <= z <= MAX_ZOOM_LIMIT:
            raise unprocessable(
                f"{label}的 z 必须是 0 到 {MAX_ZOOM_LIMIT} 之间的整数"
            )
        side = 1 << z
        if not 0 <= x < side:
            raise unprocessable(f"{label}的 x 必须在 0 到 {side - 1} 之间")
        if not 0 <= y < side:
            raise unprocessable(f"{label}的 y 必须在 0 到 {side - 1} 之间")
        if count < 1:
            raise unprocessable(f"{label}的 count 必须是正整数（>= 1）")

        key = (z, x, y)
        if key in seen:
            raise unprocessable(
                f"{label}的 z/x/y = {z}/{x}/{y} 与前面的格重复，"
                "同一格不允许重复计数"
            )
        cells.append((z, x, y, count))
        seen.add(key)

    _reject_overlapping_tiles(cells)
    return cells


def _strict_int(value: Any, field: str, label: str) -> int:
    """校验值为整数（拒绝布尔、浮点与字符串）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise unprocessable(f"{label}的 {field} 必须是整数")
    return value


def _reject_overlapping_tiles(cells: list[tuple[int, int, int, int]]) -> None:
    """拒绝混合层级格之间的正面积重叠；仅边界相接合法。

    格的实际覆盖是半开区间（见 :func:`tile_bounds`），因此两个格重叠
    当且仅当其中一个格的 z/x/y 是另一个格在较粗层级上的祖先槽：
    维护已出现格的精确槽集合与全部祖先槽集合，插入每格时双向检查即可
    （同层级格的槽要么相同——已由重复校验拦截，要么完全不相交）。
    边界恰好相接对应不同的槽，不在祖先链上，因此不会误报。
    """
    exact: set[tuple[int, int, int]] = set()
    ancestors: set[tuple[int, int, int]] = set()
    # 每个祖先槽由哪个已出现的细格占用（用于先细后粗的插入顺序报错）。
    claimed_by: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for z, x, y, _ in cells:
        key = (z, x, y)
        if key in ancestors:
            raise unprocessable(_overlap_message(key, claimed_by[key]))
        for ancestor in _ancestor_tiles(z, x, y):
            if ancestor in exact:
                raise unprocessable(_overlap_message(key, ancestor))
            ancestors.add(ancestor)
            claimed_by.setdefault(ancestor, key)
        exact.add(key)


def _ancestor_tiles(z: int, x: int, y: int) -> set[tuple[int, int, int]]:
    """返回格 (z, x, y) 在所有更粗层级上的祖先槽（不含自身）。"""
    result: set[tuple[int, int, int]] = set()
    ax, ay = x, y
    for az in range(z - 1, -1, -1):
        ax >>= 1
        ay >>= 1
        result.add((az, ax, ay))
    return result


def _overlap_message(
    one: tuple[int, int, int], other: tuple[int, int, int]
) -> str:
    """构造稳定的重叠错误信息：层级较粗的格排在前面。"""
    coarse, fine = (one, other) if one[0] < other[0] else (other, one)
    return (
        f"格 {coarse[0]}/{coarse[1]}/{coarse[2]} 与格 "
        f"{fine[0]}/{fine[1]}/{fine[2]} 在区域上重叠："
        "不同层级的发布格不得覆盖同一地理位置（仅边界相接允许）"
    )


def prepare_bounds(raw_bounds: Any) -> tuple[float, float, float, float]:
    """校验经纬矩形查询区，返回 (lat_min, lat_max, lon_min, lon_max)。

    四个边界都必须是有限数值（拒绝布尔、字符串与 NaN/无穷）；纬度在
    -90～90、经度在 -180～180 之间；且必须为正面积矩形：
    ``lat_min < lat_max``、``lon_min != lon_max``。``lon_min > lon_max``
    合法，表示查询区跨 180° 经线。
    """
    if not isinstance(raw_bounds, dict):
        raise unprocessable("bounds 必须是包含经纬矩形边界的对象")

    if set(raw_bounds) != {"lat_min", "lat_max", "lon_min", "lon_max"}:
        raise unprocessable(
            "bounds 只能包含 lat_min、lat_max、lon_min、lon_max 四个字段"
        )

    label = "bounds 的"
    lat_min = _bounded_number(raw_bounds["lat_min"], "lat_min", label, -90.0, 90.0)
    lat_max = _bounded_number(raw_bounds["lat_max"], "lat_max", label, -90.0, 90.0)
    lon_min = _bounded_number(raw_bounds["lon_min"], "lon_min", label, -180.0, 180.0)
    lon_max = _bounded_number(raw_bounds["lon_max"], "lon_max", label, -180.0, 180.0)

    if lat_min >= lat_max:
        raise unprocessable("bounds 必须满足 lat_min < lat_max（零高度矩形不构成查询区）")
    if lon_min == lon_max:
        raise unprocessable("bounds 的 lon_min 不能等于 lon_max（零宽度矩形不构成查询区）")

    return lat_min, lat_max, lon_min, lon_max


def _bounded_number(
    value: Any, field: str, label: str, low: float, high: float
) -> float:
    """校验单个边界为有限数值且在 [low, high] 内。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise unprocessable(f"{label} {field} 必须是数值")
    result = float(value)
    if not math.isfinite(result):
        raise unprocessable(f"{label} {field} 必须是有限数值")
    if not low <= result <= high:
        raise unprocessable(f"{label} {field} 必须在 {low:g} 到 {high:g} 之间")
    return result


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """返回格 z/x/y 的实际经纬覆盖 (lat_min, lat_max, lon_min, lon_max)。

    边界完全沿用 geo 模块的递归中点划分（不使用 360/2^z 反推），保证与
    ``/geo-anonymize`` 发布格边界逐位一致。覆盖区域按半开区间理解：
    经度 ``[lon_min, lon_max)``、纬度 ``[lat_min, lat_max)``，全球边界
    点 ±180、±90 并入唯一相邻的格——这正是 geo 模块“中点归北/东”规则
    对格归属的含义。因此格的覆盖在地理上互不重叠，查询矩形为闭区间时，
    仅沿边界线接触的两个格也不会被同时算作相交。
    """
    lat_min, lat_max = ROOT_LAT_MIN, ROOT_LAT_MAX
    lon_min, lon_max = ROOT_LON_MIN, ROOT_LON_MAX
    for level in range(z):
        # 从 z=1 到 z，依次取 x/y 在该层的位。geo 模块的子格编号：
        # x 位 1 为东子格，y 位 1 为南子格（y 向北为 0）。
        bit = z - 1 - level
        lat_mid = (lat_min + lat_max) / 2.0
        lon_mid = (lon_min + lon_max) / 2.0
        if (y >> bit) & 1:
            lat_max = lat_mid
        else:
            lat_min = lat_mid
        if (x >> bit) & 1:
            lon_min = lon_mid
        else:
            lon_max = lon_mid
    return lat_min, lat_max, lon_min, lon_max


def count_bounds(
    cells: list[tuple[int, int, int, int]],
    bounds: tuple[float, float, float, float],
) -> tuple[int, int, list[dict[str, Any]]]:
    """计算区域观测数的下界与上界，并给出每格的计数归属。

    返回 ``(lower_bound, upper_bound, 逐格结果)``；逐格结果与输入同序，
    每项含 z/x/y、count、归属 relation 及该格计入下界/上界的计数
    ``lower_count``/``upper_count``（inside 两界都计入，partial 只计入
    上界，outside 均为 0）。
    """
    lat_min, lat_max, lon_min, lon_max = bounds
    # lon_min > lon_max：查询区跨 180° 经线，拆成两个经度区间。
    lon_intervals = (
        [(lon_min, ROOT_LON_MAX), (ROOT_LON_MIN, lon_max)]
        if lon_min > lon_max
        else [(lon_min, lon_max)]
    )

    lower_bound = 0
    upper_bound = 0
    results: list[dict[str, Any]] = []
    for z, x, y, count in cells:
        cell_lat_min, cell_lat_max, cell_lon_min, cell_lon_max = tile_bounds(z, x, y)
        relation = _classify(
            cell_lat_min,
            cell_lat_max,
            cell_lon_min,
            cell_lon_max,
            lat_min,
            lat_max,
            lon_intervals,
        )
        lower_count = count if relation == RELATION_INSIDE else 0
        upper_count = count if relation != RELATION_OUTSIDE else 0
        lower_bound += lower_count
        upper_bound += upper_count
        results.append(
            {
                "z": z,
                "x": x,
                "y": y,
                "count": count,
                "relation": relation,
                "lower_count": lower_count,
                "upper_count": upper_count,
            }
        )

    return lower_bound, upper_bound, results


def _classify(
    cell_lat_min: float,
    cell_lat_max: float,
    cell_lon_min: float,
    cell_lon_max: float,
    query_lat_min: float,
    query_lat_max: float,
    lon_intervals: list[tuple[float, float]],
) -> str:
    """按格的半开覆盖与查询闭矩形的实际交叠判定 inside/partial/outside。"""
    # 纬度方向只有一个区间；半开覆盖对闭查询：相等端点的“接触”不构成相交。
    if cell_lat_max <= query_lat_min or cell_lat_min >= query_lat_max:
        return RELATION_OUTSIDE

    latitude_inside = cell_lat_min >= query_lat_min and cell_lat_max <= query_lat_max

    for interval_min, interval_max in lon_intervals:
        if cell_lon_max <= interval_min or cell_lon_min >= interval_max:
            continue  # 该经度区间上仅边界接触或完全分离。
        if (
            latitude_inside
            and cell_lon_min >= interval_min
            and cell_lon_max <= interval_max
        ):
            return RELATION_INSIDE
        return RELATION_PARTIAL

    return RELATION_OUTSIDE
