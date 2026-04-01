import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

# ==============================================================
# --- 1. 路径与显示范围配置 (已与WPSCF对齐) ---
# ==============================================================
TDUMP_DIR = Path(r"D:\1document\Graduation Thesis\01Code\DATA\HYSPLIT_Results")
CSV_PATH = Path(
    r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\PM2.5_Meteorology_Study_3day\Refined_Golden_Cases_Selection.csv")
SAVE_DIR = TDUMP_DIR / "Plots_Professional"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PROVINCE_SHP = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\中国_省\中国_省2.shp"
CITY_SHP = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\杭州市_市\杭州市_市.shp"

# 💡 [关键点] 必须与WPSCF图的范围完全一致
MAP_EXTENT = [105, 126, 25, 46]
HZ_LAT, HZ_LON = 30.27, 120.15

HEIGHT_COLORS = {100: '#32CD32', 500: '#1E90FF', 1000: '#FF4500'}

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================
# --- 2. 核心功能组件 ---
# ==============================================================

def read_tdump(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data_start_idx = 0
    for i, line in enumerate(lines):
        if 'PRESSURE' in line or 'HEIGHT' in line:
            data_start_idx = i + 1
            break
    traj_data = []
    for line in lines[data_start_idx:]:
        parts = line.split()
        if len(parts) >= 11:
            traj_data.append([float(parts[10]), float(parts[9])])
    return pd.DataFrame(traj_data, columns=['lon', 'lat'])


def get_clipped_features(extent):
    print("🌍 正在同步行政边界底图...")
    # 为了边缘显示完整，裁剪范围略微扩大
    bbox = box(extent[0] - 2, extent[2] - 2, extent[1] + 2, extent[3] + 2)
    gdf_prov = gpd.read_file(PROVINCE_SHP, encoding='utf-8')
    clipped_prov = gpd.clip(gdf_prov, bbox)
    gdf_city = gpd.read_file(CITY_SHP, encoding='utf-8')
    clipped_city = gpd.clip(gdf_city, bbox)

    prov_feat = cfeature.ShapelyFeature(clipped_prov.geometry, ccrs.PlateCarree(),
                                        edgecolor='#666666', facecolor='#FDFDFD', linewidth=0.7)
    city_feat = cfeature.ShapelyFeature(clipped_city.geometry, ccrs.PlateCarree(),
                                        edgecolor='#1A237E', facecolor='none', linewidth=1.5)
    return prov_feat, city_feat


# ==============================================================
# --- 3. 绘图主引擎 ---
# ==============================================================

def plot_trajectories_aligned():
    df_cases = pd.read_csv(CSV_PATH)
    prov_feat, city_feat = get_clipped_features(MAP_EXTENT)

    for case_idx, row in df_cases.iterrows():
        start_date = pd.to_datetime(row['分析开始']).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(row['分析结束']).strftime('%Y-%m-%d')
        time_period = f"{start_date} 至 {end_date}"
        case_tag = row['冬季案例']

        print(f"🎬 正在绘制对齐版 Case_{case_idx} ...")

        # 1. 保持画布比例 (10, 10)
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # 💡 [关键点1] 强制设定地图在画布中的坐标位置 [左, 下, 宽, 高]
        # 必须与 WPSCF 脚本中的数值完全一致
        ax.set_position([0.05, 0.05, 0.8, 0.85])

        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

        # 添加底图
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#E3F2FD', zorder=0)
        ax.add_feature(prov_feat, zorder=1)
        ax.add_feature(city_feat, zorder=2)

        has_traj = False
        for h, color in HEIGHT_COLORS.items():
            file_name = f"tdump_{case_idx}_H{h}"
            file_path = TDUMP_DIR / file_name

            if file_path.exists():
                df = read_tdump(file_path)
                ax.plot(df['lon'], df['lat'], color=color, linewidth=2.5,
                        transform=ccrs.PlateCarree(), label=f'高度: {h}m', zorder=5)
                # 杭州受体点
                ax.plot(HZ_LON, HZ_LAT, marker='*', color='gold', markersize=16,
                        markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
                has_traj = True

        if not has_traj:
            plt.close();
            continue

        # 设置双行标题
        plt.title(f"HYSPLIT 72h 后向轨迹分析 ({case_tag})\n时段: {time_period}",
                  fontsize=16, fontweight='bold', pad=15)

        ax.legend(loc='lower left', frameon=True, shadow=True, title="起始高度")

        # 💡 [关键点2] 固定刻度间隔为 3 度
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.4, color='gray', zorder=4)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(105, 127, 3))
        gl.ylocator = mticker.FixedLocator(np.arange(25, 47, 3))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # 💡 [关键点3] 保存时取消 bbox_inches='tight'，避免破坏比例
        output_file = SAVE_DIR / f"Trajectory_Case_{case_idx}_Aligned.png"
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"✅ 对齐图件已保存: {output_file.name}")


if __name__ == "__main__":
    plot_trajectories_aligned()
    print("\n✨ 轨迹对齐任务完成！请使用这两个脚本生成的图进行组图对比。")