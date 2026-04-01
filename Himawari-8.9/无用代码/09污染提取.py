import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import re
from shapely.geometry import box
import concurrent.futures

# --- 配置 ---
LON_MIN, LON_MAX = 114.0, 123.0
LAT_MIN, LAT_MAX = 27.0, 36.0
INPUT_ROOT = Path(r"E:\PM25_Retrieval_Results")
SHP_PATH = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\中国_市\中国_市2.shp"
OUTPUT_CSV = r"E:\PM2.5_Pollution\Full_Hourly_Data.csv"
CITY_NAME_COL = 'name'

def get_time(filename):
    match = re.search(r'(202[0-3])(\d{2})(\d{2})_(\d{2})', filename)
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:00" if match else None

def process_single_file(f_path, city_masks):
    time_str = get_time(f_path.name)
    if not time_str: return None
    try:
        with rasterio.open(f_path) as src:
            data = src.read(1)
            data = np.where((data < 0) | (data > 1000), np.nan, data)
            row = {'Time': time_str}
            for city_name, mask in city_masks.items():
                vals = data[mask]
                row[city_name] = np.nanmean(vals) if vals.size > 0 else 0
            return row
    except: return None

def main():
    Path(r"E:\Migration_Analysis_Pro").mkdir(parents=True, exist_ok=True)
    print("🎯 初始化空间掩膜...")
    all_files = sorted(list(INPUT_ROOT.rglob("*.tif")))
    with rasterio.open(all_files[0]) as src:
        transform, shape, tif_crs = src.transform, src.shape, src.crs

    try:
        china_gdf = gpd.read_file(SHP_PATH, encoding='utf-8')
    except:
        china_gdf = gpd.read_file(SHP_PATH, encoding='gbk')

    if china_gdf.crs != tif_crs: china_gdf = china_gdf.to_crs(tif_crs)
    cities_gdf = china_gdf[china_gdf.intersects(box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX))].copy()
    cities_gdf = cities_gdf.dissolve(by=CITY_NAME_COL).reset_index()

    city_masks = {row[CITY_NAME_COL]: (rasterize([(row.geometry, 1)], out_shape=shape, transform=transform, fill=0) == 1)
                  for _, row in cities_gdf.iterrows()}

    print(f"🚀 提取 {len(all_files)} 个文件的浓度数据...")
    final_data = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, f, city_masks) for f in all_files]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            if f.result(): final_data.append(f.result())

    df = pd.DataFrame(final_data)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)



    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✨ 提取完成！数据已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()