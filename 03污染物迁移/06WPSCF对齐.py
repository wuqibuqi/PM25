import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box

# ==============================================================
# --- 1. 基础路径配置 ---
# ==============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

TDUMP_DIR = Path(r"D:\1document\Graduation Thesis\01Code\DATA\HYSPLIT_Results")
PM25_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM25_Retrieval_Results_Seamless")
CSV_PATH = Path(
    r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\PM2.5_Meteorology_Study_3day\Refined_Golden_Cases_Selection.csv")
SAVE_DIR = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\Source_Analysis_Final")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PROVINCE_SHP = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\中国_省\中国_省2.shp"
CITY_SHP = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\杭州市_市\杭州市_市.shp"

HZ_LAT, HZ_LON = 30.27, 120.15
MAP_EXTENT = [105, 126, 25, 46]


# ==============================================================
# --- 2. 核心算法 ---
# ==============================================================

def get_pm25_val(t):
    sub = PM25_ROOT / t.strftime('%Y') / t.strftime('%m') / t.strftime('%d')
    f = sub / f"PM25_Seamless_{t.strftime('%Y%m%d_%H')}00.tif"
    if not f.exists(): return None
    with rasterio.open(f) as src:
        r, c = src.index(HZ_LON, HZ_LAT)
        data = src.read(1)
        return data[r, c] if data[r, c] > 0 else None


def read_traj_coords(f_path):
    with open(f_path, 'r') as f:
        lines = f.readlines()
    start = 0
    for i, line in enumerate(lines):
        if 'PRESSURE' in line or 'HEIGHT' in line:
            start = i + 1;
            break
    return [[float(l.split()[10]), float(l.split()[9])] for l in lines[start:] if len(l.split()) >= 11]


# ==============================================================
# --- 3. 绘图主程序 ---
# ==============================================================

def run_pscf_multi_height_fixed():
    df_cases = pd.read_csv(CSV_PATH)

    # 预载底图
    bbox = box(MAP_EXTENT[0] - 2, MAP_EXTENT[2] - 2, MAP_EXTENT[1] + 2, MAP_EXTENT[3] + 2)
    gdf_prov = gpd.read_file(PROVINCE_SHP).clip(bbox)
    prov_feat = cfeature.ShapelyFeature(gdf_prov.geometry, ccrs.PlateCarree(),
                                        edgecolor='#666666', facecolor='#FDFDFD', lw=0.7)
    gdf_city = gpd.read_file(CITY_SHP).clip(bbox)
    city_feat = cfeature.ShapelyFeature(gdf_city.geometry, ccrs.PlateCarree(),
                                        edgecolor='#1A237E', facecolor='none', lw=1.5)

    for idx, row in df_cases.iterrows():
        case_tag = row['冬季案例']
        peak_t = pd.to_datetime(row['峰值时刻'])
        hz_pm = get_pm25_val(peak_t)
        if hz_pm is None: continue

        DYNAMIC_THRESHOLD = 70 if hz_pm >= 75 else 50
        print(f"🎬 综合分析 Case {idx} (多高度融合) | 浓度: {hz_pm:.1f}")

        # 网格统计逻辑 (保持不变)
        lon_bins = np.arange(100, 131, 0.5)
        lat_bins = np.arange(20, 51, 0.5)
        n_ij = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
        m_ij = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))

        heights = [100, 500, 1000]
        found_any = False
        for h in heights:
            t_file = TDUMP_DIR / f"tdump_{idx}_H{h}"
            if t_file.exists():
                found_any = True
                coords = read_traj_coords(t_file)
                for ln, lt in coords:
                    if 100 <= ln < 130 and 20 <= lt < 50:
                        lt_idx = np.searchsorted(lat_bins, lt) - 1
                        ln_idx = np.searchsorted(lon_bins, ln) - 1
                        n_ij[lt_idx, ln_idx] += 1
                        if hz_pm > DYNAMIC_THRESHOLD:
                            m_ij[lt_idx, ln_idx] += 1

        if not found_any: continue

        pscf = np.divide(m_ij, n_ij, out=np.zeros_like(m_ij), where=n_ij > 0)
        avg_n = np.mean(n_ij[n_ij > 0])
        w = np.ones_like(n_ij)
        w[n_ij < avg_n * 0.7] = 0.7;
        w[n_ij < avg_n * 0.5] = 0.5;
        w[n_ij < avg_n * 0.2] = 0.1
        wpscf = pscf * w
        wpscf_masked = np.ma.masked_where(wpscf <= 0.05, wpscf)

        # --- 核心修改：对齐绘图区 ---
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # 💡 [关键点1]：强制指定地图轴在画布中的位置 [左, 下, 宽, 高]
        # 这个位置要和轨迹图的代码完全一致
        ax.set_position([0.05, 0.05, 0.8, 0.85])

        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#E3F2FD', zorder=0)
        ax.add_feature(prov_feat, zorder=1)
        ax.add_feature(city_feat, zorder=2)

        mesh = ax.pcolormesh(lon_bins, lat_bins, wpscf_masked, cmap='YlOrRd',
                             transform=ccrs.PlateCarree(), vmax=0.7, zorder=3, alpha=0.8)

        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.4, color='gray', zorder=4)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(105, 127, 3))
        gl.ylocator = mticker.FixedLocator(np.arange(25, 47, 3))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        ax.plot(HZ_LON, HZ_LAT, '*', color='gold', markersize=16, markeredgecolor='black', transform=ccrs.PlateCarree(),
                zorder=10)

        plt.title(f"杭州冬季 $PM_{{2.5}}$ 综合潜在源区解析 ({case_tag})\n" +
                  f"基于 100/500/1000m 融合轨迹", fontsize=14, fontweight='bold', pad=15)

        # 💡 [关键点2]：手动创建色柱轴，不让它挤压地图
        cax = fig.add_axes([0.9, 0.2, 0.03, 0.55])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label('WPSCF 综合贡献强度指标', fontsize=12)

        # 💡 [关键点3]：保存时不使用 bbox_inches='tight'，防止重新裁剪破坏比例
        plt.savefig(SAVE_DIR / f"WPSCF_Case_{idx}_Aligned.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    run_pscf_multi_height_fixed()