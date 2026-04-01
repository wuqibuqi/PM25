import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from datetime import timedelta
import os
import re

# ==============================
# --- 1. 基础路径配置 ---
# ==============================
# 请确保以下路径指向您的数据存放位置
PM25_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM25_Retrieval_Results_Seamless")
U10_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\ERA5_TIF\u10")
V10_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\ERA5_TIF\v10")
CASES_CSV = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\PM2.5_Meteorology_Study_3day\Refined_Golden_Cases_Selection.csv")
OUT_DIR = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\PM2.5_Meteorology_Study_3day\Evolution_Real_Wind_3day_12grid")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITY_SHP_PATH = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\杭州市_市\杭州市_市.shp"
PROVINCE_SHP_PATH = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\中国_省\中国_省2.shp"

# 研究区域范围 (长三角/杭州周边)
LON_MIN, LON_MAX = 114.0, 123.0
LAT_MIN, LAT_MAX = 27.0, 36.0

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================
# --- 2. 核心功能函数 ---
# ==============================

def build_file_index(root_path, suffix_key=""):
    """建立快速时间戳索引字典"""
    index = {}
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.endswith(".tif") and suffix_key in f:
                match = re.search(r'(\d{8}_\d{2})', f)
                if match:
                    index[match.group(1)] = os.path.join(root, f)
    return index


def read_raster_and_coords(tif_path):
    """读取栅格数据并生成匹配的地理坐标网格"""
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        data[data == src.nodata] = np.nan

        bounds = src.bounds
        # 经度从西向东，纬度从高到低（Top-Down 顺序）
        lons = np.linspace(bounds.left, bounds.right, src.width)
        lats = np.linspace(bounds.top, bounds.bottom, src.height)

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        return data, lon_grid, lat_grid


def plot_case_matrix_12grid(case_row, clipped_prov, clipped_hz, pm_index, u_index, v_index):
    """绘制 3x4 布局的 12 宫格演变矩阵"""
    case_name = case_row['冬季案例']
    start_t = pd.to_datetime(case_row['分析开始'])
    end_t = pd.to_datetime(case_row['分析结束'])

    # 将 72 小时平分为 12 个时间点 (每 6 小时一跳)
    target_times = [start_t + timedelta(seconds=s) for s in np.linspace(0, (end_t - start_t).total_seconds(), 12)]

    fig, axes = plt.subplots(3, 4, figsize=(22, 16), constrained_layout=True)
    axes = axes.flatten()

    im = None
    q = None

    for i, t in enumerate(target_times):
        ax = axes[i]
        t_key = t.strftime('%Y%m%d_%H')
        pm_path, u_path, v_path = pm_index.get(t_key), u_index.get(t_key), v_index.get(t_key)

        # 1. 绘制 PM2.5 浓度底图
        if pm_path:
            with rasterio.open(pm_path) as src:
                pm_data = src.read(1).astype(np.float32)
                pm_data[(pm_data < 0) | (pm_data > 800)] = np.nan
                im = ax.imshow(pm_data, cmap='jet', vmin=0, vmax=150,
                               extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], origin='upper')

        # 2. 绘制风场箭头
        if u_path and v_path:
            u_data, lon_g, lat_g = read_raster_and_coords(u_path)
            v_data, _, _ = read_raster_and_coords(v_path)

            # 物理纠偏：在 Top-Down 投影中，可能需要反转 V 符号以指向东南
            # v_plot = -v_data

            # 抽样间隔：由于 12 宫格子图较小，skip 设为 6-8 以保持清晰
            skip = 5
            q = ax.quiver(lon_g[::skip, ::skip], lat_g[::skip, ::skip],
                          u_data[::skip, ::skip], v_data[::skip, ::skip],
                          color='white', scale=45, width=0.004, alpha=0.9, zorder=20, pivot='middle')

        # 3. 叠加行政边界
        if not clipped_prov.empty:
            clipped_prov.plot(ax=ax, linewidth=0.6, edgecolor='#333333', facecolor='none', alpha=0.5, zorder=2)
        clipped_hz.plot(ax=ax, linewidth=1.8, edgecolor='darkblue', facecolor='none', zorder=10)

        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_title(f"{t.strftime('%m-%d %H:00')}", fontsize=14, fontweight='bold')
        ax.set_axis_off()

    # 添加统一的色标和风速参考
    if im:
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.6, aspect=25, pad=0.02)
        cbar.set_label('PM2.5 浓度 ($\mu g/m^3$)', fontsize=14)
        if q:
            axes[-1].quiverkey(q, 0.92, 0.05, 10, r'$10 \ m/s$', labelpos='E', coordinates='figure', color='black')

    # plt.suptitle(f"长三角 PM2.5 迁移演变序列 (72h 窗口 / 6h 采样)：{case_name}", fontsize=24, fontweight='bold', y=0.98)

    save_path = OUT_DIR / f"Evolution_12Grid_{case_name.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 渲染完成: {case_name}")


def main():
    # 建立全局索引
    pm_index = build_file_index(PM25_ROOT)
    u_index = build_file_index(U10_ROOT, "u10")
    v_index = build_file_index(V10_ROOT, "v10")
    print(f"📊 索引统计: PM25({len(pm_index)}), U10({len(u_index)}), V10({len(v_index)})")

    # 准备裁剪后的边界
    research_bbox = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    try:
        prov = gpd.read_file(PROVINCE_SHP_PATH, encoding='utf-8')
        clipped_prov = gpd.clip(prov, research_bbox)
        cities = gpd.read_file(CITY_SHP_PATH, encoding='utf-8')
        clipped_hz = gpd.clip(cities[cities['name'].str.contains('杭州')], research_bbox)
    except Exception as e:
        print(f"❌ SHP 载入失败: {e}")
        return

    # 执行绘图
    cases_df = pd.read_csv(CASES_CSV)
    for _, row in cases_df.iterrows():
        plot_case_matrix_12grid(row, clipped_prov, clipped_hz, pm_index, u_index, v_index)

    print(f"✨ 任务结束。所有案例的高频演变矩阵已生成至: {OUT_DIR}")


if __name__ == "__main__":
    main()