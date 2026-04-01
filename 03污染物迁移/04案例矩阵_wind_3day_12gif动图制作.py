import os

# 锁定底层环境，防止 Intel Fortran 冲突
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import re
from tqdm import tqdm
import imageio.v2 as imageio

# ==============================================================
# --- 1. 基础路径与反演控制配置 ---
# ==============================================================
PM25_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM25_Retrieval_Results_Seamless")
U10_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\ERA5_TIF\u10")
V10_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\ERA5_TIF\v10")
CASES_CSV = Path(
    r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\PM2.5_Meteorology_Study_3day\Refined_Golden_Cases_Selection.csv")

# --- [反演控制开关] ---
OVERWRITE = True  # True: 强制覆盖已有文件 | False: 断点续传 (跳过已生成的图片和GIF)

# --- [输出目录] ---
AUTO_OUT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\Auto_Worst_Case_Dynamic")
MANUAL_OUT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\Manual_Custom_Dynamic")

# --- [手动查询参数] ---
MANUAL_START = "2023-12-27 00:00"
MANUAL_END = "2023-12-30 23:00"
MANUAL_STEP = 1

# 行政边界
CITY_SHP_PATH = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\杭州市_市\杭州市_市.shp"
PROVINCE_SHP_PATH = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\中国_省\中国_省2.shp"

# 研究区域范围
LON_MIN, LON_MAX = 114.0, 123.0
LAT_MIN, LAT_MAX = 27.0, 36.0

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================
# --- 2. 核心功能函数 ---
# ==============================================================

def build_file_index(root_path, key=""):
    idx = {}
    for root, _, files in os.walk(root_path):
        for f in files:
            if f.endswith(".tif") and key in f:
                match = re.search(r'(\d{8}_\d{2})', f)
                if match: idx[match.group(1)] = os.path.join(root, f)
    return idx


def get_sequence_vmax(time_range, pm_idx):
    """预扫描量程"""
    max_vals = []
    for t in time_range:
        t_key = t.strftime('%Y%m%d_%H')
        path = pm_idx.get(t_key)
        if path:
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                data[(data < 0) | (data > 1000)] = np.nan
                if np.any(~np.isnan(data)):
                    max_vals.append(np.nanpercentile(data, 99.5))
    return int(np.ceil(max(max_vals) / 10.0)) * 10 if max_vals else 150


def render_sequence(time_range, output_path, title_prefix, pm_idx, u_idx, v_idx, prov, hz):
    """带断点续传功能的渲染引擎"""
    output_path.mkdir(parents=True, exist_ok=True)
    gif_path = output_path / f"{output_path.name}_Evolution.gif"

    # --- [续传逻辑 1: GIF 级别] ---
    if not OVERWRITE and gif_path.exists():
        print(f"⏭️  检测到 GIF 已存在，跳过任务: {output_path.name}")
        return

    temp_dir = output_path / "temp_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    vmax_val = get_sequence_vmax(time_range, pm_idx)
    print(f"🎨 动态量程锁定: 0 - {vmax_val} μg/m³")

    frame_files = []
    for i, t in enumerate(tqdm(time_range, desc=f"🎬 渲染 {output_path.name}")):
        t_key = t.strftime('%Y%m%d_%H')
        f_path = temp_dir / f"frame_{i:03d}_{t_key}.png"

        # --- [续传逻辑 2: 帧级别] ---
        if not OVERWRITE and f_path.exists():
            frame_files.append(f_path)
            continue

        pm_p, u_p, v_p = pm_idx.get(t_key), u_idx.get(t_key), v_idx.get(t_key)
        if not pm_p: continue

        fig, ax = plt.subplots(figsize=(12, 10))

        # 1. PM2.5 底图
        with rasterio.open(pm_p) as src:
            data = src.read(1).astype(np.float32)
            data[(data < 0) | (data > vmax_val * 1.5)] = np.nan
            im = ax.imshow(data, cmap='jet', vmin=0, vmax=150,#更改图例最值
                           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], origin='upper')

        # 2. 风场
        if u_p and v_p:
            with rasterio.open(u_p) as su, rasterio.open(v_p) as sv:
                u, v = su.read(1), sv.read(1)
                lons = np.linspace(su.bounds.left, su.bounds.right, su.width)
                lats = np.linspace(su.bounds.top, su.bounds.bottom, su.height)
                lon_g, lat_g = np.meshgrid(lons, lats)
                skip = 6
                ax.quiver(lon_g[::skip, ::skip], lat_g[::skip, ::skip],
                          u[::skip, ::skip], v[::skip, ::skip],
                          color='white', scale=50, width=0.003, alpha=0.7, pivot='middle')

        # 3. 边界与修饰
        prov.plot(ax=ax, linewidth=0.8, edgecolor='#333333', facecolor='none', alpha=0.6)
        hz.plot(ax=ax, linewidth=2.5, edgecolor='darkblue', facecolor='none', zorder=10)

        ax.set_title(f"{title_prefix}\n{t.strftime('%Y-%m-%d %H:00')}", fontsize=16, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('PM2.5 浓度 ($\mu g/m^3$)', fontsize=12)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)

        plt.savefig(f_path, dpi=120, bbox_inches='tight')
        plt.close()
        frame_files.append(f_path)

    # 4. 合成 GIF
    if frame_files:
        print(f"📽️  正在合成动画: {gif_path.name}...")
        with imageio.get_writer(gif_path, mode='I', duration=0.7) as writer:
            for f in sorted(frame_files):  # 确保帧顺序正确
                writer.append_data(imageio.imread(f))
        print(f"✅ 任务圆满完成！")

        # 只有合成成功后才清理临时文件
        # for f in frame_files: os.remove(f)


# ==============================================================
# --- 3. 执行入口 ---
# ==============================================================

def main():
    pm_idx = build_file_index(PM25_ROOT)
    u_idx = build_file_index(U10_ROOT, "u10")
    v_idx = build_file_index(V10_ROOT, "v10")

    bbox = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    prov_gdf = gpd.read_file(PROVINCE_SHP_PATH).clip(bbox)
    hz_gdf = gpd.read_file(CITY_SHP_PATH).clip(bbox)

    # --- 引擎 A: 自动模式 ---
    if CASES_CSV.exists():
        df = pd.read_csv(CASES_CSV)
        worst = df.loc[df['最高浓度'].idxmax()]
        auto_range = pd.date_range(start=worst['分析开始'], end=worst['分析结束'], freq='1h')
        render_sequence(auto_range, AUTO_OUT, f"自动识别最严重案例: {worst['冬季案例']}",
                        pm_idx, u_idx, v_idx, prov_gdf, hz_gdf)

    # --- 引擎 B: 手动模式 ---
    manual_range = pd.date_range(start=MANUAL_START, end=MANUAL_END, freq=f'{MANUAL_STEP}h')
    render_sequence(manual_range, MANUAL_OUT, f"手动查询序列({MANUAL_STEP}h间隔)",
                    pm_idx=pm_idx, u_idx=u_idx, v_idx=v_idx, prov=prov_gdf, hz=hz_gdf)


if __name__ == "__main__":
    main()