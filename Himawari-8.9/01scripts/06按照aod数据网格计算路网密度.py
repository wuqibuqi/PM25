import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import os
import time

# --- 1. 路径配置 ---
AOD_TEMPLATE = r"E:\Himawari-8_TIFF\202204\11\H08_20220411_0000_1HARP031_FLDK.02401_02401.tif"  # 你的 181x181 模板
ROAD_SHP_PATH = r"E:\Static_Features\Roads\china-230101-free.shp\gis_osm_roads_free_1.shp"
OUTPUT_TIF = r"E:\Static_Features\Roads\Road_Density_5km_Forced_181.tif"


def calculate_density_forced_align():
    if not os.path.exists(AOD_TEMPLATE):
        print("❌ 找不到 AOD 模板文件，请检查路径")
        return

    # 1. 读取 AOD 模板的元数据（这是灵魂步骤）
    with rasterio.open(AOD_TEMPLATE) as src:
        meta = src.meta.copy()
        height = src.height  # 应该是 181
        width = src.width  # 应该是 181
        transform = src.transform
        crs = src.crs
        # 获取左上角和分辨率
        res_x = transform[0]
        res_y = transform[4]  # 通常是负数

    print(f"📐 模板读取成功: {height}x{width}, 分辨率: {res_x}")

    # 2. 读取矢量路网
    # 获取模板的边界范围
    with rasterio.open(AOD_TEMPLATE) as src:
        west, south, east, north = src.bounds

    print("📖 正在读取并裁切矢量路网...")
    roi_box = box(west, south, east, north)
    roads = gpd.read_file(ROAD_SHP_PATH, bbox=roi_box)
    main_roads = roads[roads['fclass'].isin(['motorway', 'trunk', 'primary', 'secondary'])].copy()
    roads_metric = main_roads.to_crs(epsg=3857)
    sindex = roads_metric.sindex

    # 3. 开始计算
    density_map = np.zeros((height, width), dtype=np.float32)
    start_time = time.time()

    print(f"📐 正在严格按照模板生成网格密度...")
    for r in range(height):
        if r % 20 == 0: print(f"进度: {r}/{height}")
        for c in range(width):
            # 直接使用模板的像素坐标转换工具得到该像素的边界
            # xy(r, c) 得到像素中心，但我们需要像素的 box
            # 这里的计算方式确保了空间上的绝对重合
            lon_center, lat_center = rasterio.transform.xy(transform, r, c)

            # 构建像素波形的矩形 (以像素中心向四周推半个分辨率)
            half_x = abs(res_x) / 2
            half_y = abs(res_y) / 2
            cell_box = box(lon_center - half_x, lat_center - half_y,
                           lon_center + half_x, lat_center + half_y)

            # 投影到米进行计算
            cell_gdf = gpd.GeoSeries([cell_box], crs=4326).to_crs(epsg=3857)
            cell_geom = cell_gdf.iloc[0]

            possible_matches = list(sindex.intersection(cell_geom.bounds))
            if possible_matches:
                intersecting_roads = roads_metric.iloc[possible_matches]
                lengths = intersecting_roads.intersection(cell_geom).length
                density_map[r, c] = lengths.sum() / 1000.0

    # 4. 严格按照模板元数据保存
    meta.update({
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "compress": "lzw"
    })

    with rasterio.open(OUTPUT_TIF, 'w', **meta) as dst:
        dst.write(density_map, 1)

    print(f"🎉 强制对齐完成！输出文件: {OUTPUT_TIF}，形状: {height}x{width}")


if __name__ == "__main__":
    calculate_density_forced_align()