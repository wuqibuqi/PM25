import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import re
import concurrent.futures
import matplotlib.pyplot as plt

# --- 基础配置 ---
INPUT_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM25_Retrieval_Results_Seamless")
SHP_PATH = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\中国_市\中国_市2.shp"
OUTPUT_DIR = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CITY = "杭州市"


def get_time(filename):
    # 匹配文件名中的时间: 202x MM DD _ HH
    match = re.search(r'(202[0-4])(\d{2})(\d{2})_(\d{2})', filename)
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:00" if match else None


def process_file(f_path, mask):
    time_str = get_time(f_path.name)
    if not time_str: return None
    try:
        with rasterio.open(f_path) as src:
            data = src.read(1).astype(np.float32)
            # 气象学数据清洗：剔除无效值和异常高值
            data[(data < 0) | (data > 1000)] = np.nan
            vals = data[mask]
            # 计算区域平均值
            mean_val = np.nanmean(vals) if vals.size > 0 else np.nan
            return {'Time': time_str, 'HZ_Mean': mean_val}
    except:
        return None


def main():
    print(f"🌍 正在初始化 {TARGET_CITY} 的空间掩膜...")
    all_files = sorted(list(INPUT_ROOT.rglob("*.tif")))

    # 1. 准备杭州的掩膜
    with rasterio.open(all_files[0]) as src:
        meta = src.meta
        hz_gdf = gpd.read_file(SHP_PATH, encoding='utf-8')
        hz_geometry = hz_gdf[hz_gdf['name'] == TARGET_CITY].geometry
        hz_mask = rasterize(hz_geometry, out_shape=src.shape, transform=src.transform, fill=0, default_value=1,
                            all_touched=True).astype(bool)

    # 2. 并行提取均值
    print(f"🚀 开始提取 {len(all_files)} 小时的杭州均值数据...")
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, f, hz_mask) for f in all_files]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            if f.result(): results.append(f.result())

    # 3. 整理并生成时序图
    df = pd.DataFrame(results)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').set_index('Time')
    df.to_csv(OUTPUT_DIR / "Hangzhou_Hourly_Mean.csv")

    print(f"📈 正在生成全时序趋势图...")
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['HZ_Mean'], color='#2c3e50', linewidth=0.5, alpha=0.8)
    plt.axhline(y=75, color='red', linestyle='--', label='二级标准 (75)')
    plt.title(f"{TARGET_CITY} PM2.5 逐小时均值演变序列 (2020-2023)")
    plt.ylabel("浓度 ($\mu g/m^3$)")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "Hangzhou_Timeline_Full.png", dpi=300)

    print(f"✨ 提取完成！请在 {OUTPUT_DIR} 查看 CSV 和趋势图。")


if __name__ == "__main__":
    main()