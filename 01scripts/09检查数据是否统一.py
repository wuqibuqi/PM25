import os
import rasterio
import pandas as pd
import numpy as np
from datetime import datetime

# --- 1. 定义数据资产字典 (严格匹配你的路径) ---
DATA_ASSETS = {
    "CLCD_TIF": r"E:\CLCD_Data\CLCD_WGS84\CLCD_ROI_2020.tif",
    "DEM": r"E:\DEM\ChangSanJiao_DEM_30m.tif",
    "Pop": r"E:\Static_Features\Population\ROI_Pop_2020_100m.tif",
    "Road_Density": r"E:\Static_Features\Roads\Road_Density_1km_Local.tif",
    "NDVI_Sample": r"E:\Static_Features\NDVI_Daily\NDVI_20200101.tif",  # 抽样一张
    "AOD_Sample": r"E:\Himawari-8_TIFF\202301\16\H09_20230116_0200_1HARP031_FLDK.02401_02401.tif",  # 请确认一个存在的路径
    "ERA5_Sample": r"E:\ERA5_TIF\blh\2020\01\ERA5_20200101_0000_blh.tif",  # 请确认一个存在的路径
}

# 基准参数 (1km 分辨率)
BASE_RES = 0.01  # 约 1km
BASE_CRS = "EPSG:4326"


def analyze_assets():
    print(f"{'数据项':<20} | {'坐标系':<10} | {'分辨率(deg)':<15} | {'行列号(H,W)':<15} | {'状态'}")
    print("-" * 85)

    report = []

    for name, path in DATA_ASSETS.items():
        status = "✅ OK"
        if not os.path.exists(path):
            print(f"{name:<20} | {'文件不存在':<60}")
            continue

        with rasterio.open(path) as src:
            crs = str(src.crs)
            res_x, res_y = src.res
            height, width = src.shape

            # 逻辑检查
            issues = []
            if BASE_CRS not in crs:
                issues.append("投影不匹配")
                status = "❌ 需重投影"

            # 分辨率检查 (允许 10% 的误差范围)
            if not (0.9 * BASE_RES <= abs(res_x) <= 1.1 * BASE_RES):
                issues.append(f"Res:{abs(res_x):.4f}")
                status = "⚠️ 需重采样"

            print(f"{name:<20} | {crs[:10]:<10} | {abs(res_x):.6f} | {height:>5}x{width:<8} | {status}")

            report.append({
                "Name": name, "CRS": crs, "Res": abs(res_x),
                "H": height, "W": width, "Issues": ", ".join(issues)
            })

    return pd.DataFrame(report)


def check_csv_alignment():
    print("\n📊 --- CSV 数据格式核查 ---")
    pm25_path = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\Output\Ground_Value_2020_2023.csv"
    clcd_csv_path = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\Output\CLCD_Full_LC.csv"

    for p in [pm25_path, clcd_csv_path]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"文件: {os.path.basename(p)}")
            print(f"  - 数据行数: {len(df)}")
            print(f"  - 列名: {list(df.columns)}")
        else:
            print(f"❌ 找不到 CSV: {p}")


if __name__ == "__main__":
    df_report = analyze_assets()
    check_csv_alignment()