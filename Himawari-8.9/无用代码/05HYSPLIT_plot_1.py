import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

# ==============================================================
# --- 1. 路径与显示范围配置 ---
# ==============================================================
TDUMP_DIR = Path(r"D:\1document\Graduation Thesis\01Code\DATA\HYSPLIT_Results")
CSV_PATH = Path(
    r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\PM2.5_Meteorology_Study_3day\Refined_Golden_Cases_Selection.csv")
SAVE_DIR = TDUMP_DIR / "Plots_Professional"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PROVINCE_SHP = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\中国_省\中国_省2.shp"
CITY_SHP = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\杭州市_市\杭州市_市.shp"

# 绘图显示范围 (建议宽一点以看清来源)
MAP_EXTENT = [100, 130, 22, 50]
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
    print("🌍 正在裁剪行政边界底图...")
    bbox = box(extent[0], extent[2], extent[1], extent[3])
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
# --- 3. 绘图主引擎 (新增动态标题逻辑) ---
# ==============================================================

def plot_trajectories():
    # 0. 读取 CSV 获取时间信息
    df_cases = pd.read_csv(CSV_PATH)
    prov_feat, city_feat = get_clipped_features(MAP_EXTENT)

    for case_idx, row in df_cases.iterrows():
        # --- [新增] 提取并格式化时间段 ---
        start_date = pd.to_datetime(row['分析开始']).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(row['分析结束']).strftime('%Y-%m-%d')
        time_period = f"{start_date} 至 {end_date}"
        case_tag = row['冬季案例']

        print(f"🎬 正在绘制 Case_{case_idx} ({time_period}) ...")

        fig = plt.figure(figsize=(10, 11))  # 略微增加高度以容纳双行标题
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

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
                ax.plot(HZ_LON, HZ_LAT, marker='*', color='gold', markersize=12,
                        markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
                has_traj = True

        if not has_traj:
            plt.close();
            continue

        # --- [优化] 设置双行标题：第一行是案例名，第二行是具体时间段 ---
        plt.title(f"HYSPLIT 72h 后向轨迹分析 ({case_tag})\n时段: {time_period}",
                  fontsize=16, fontweight='bold', pad=15)

        ax.legend(loc='lower left', frameon=True, shadow=True, title="起始高度")

        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                          linestyle='--', alpha=0.4, color='gray')
        gl.top_labels = False
        gl.right_labels = False

        output_file = SAVE_DIR / f"Trajectory_Case_{case_idx}_Final.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存成功: {output_file.name}")


if __name__ == "__main__":
    plot_trajectories()