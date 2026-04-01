import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from datetime import timedelta
import os

# ==============================
# --- 1. 基础路径配置 ---
# ==============================
PM25_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM25_Retrieval_Results_Seamless")
U10_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\ERA5_TIF\u10")
V10_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\ERA5_TIF\v10")
CASES_CSV = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\PM2.5_Meteorology_Study_3day\Refined_Golden_Cases_Selection.csv")
OUT_DIR = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\PM2.5_Meteorology_Study_3day\Evolution_Real_Wind_3day")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITY_SHP_PATH = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\杭州市_市\杭州市_市.shp"
PROVINCE_SHP_PATH = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\中国_省\中国_省2.shp"

LON_MIN, LON_MAX = 114.0, 123.0
LAT_MIN, LAT_MAX = 27.0, 36.0

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================
# --- 2. 【核心优化】预索引功能 ---
# ==============================
def build_file_index(root_path, suffix_key=""):
    """
    遍历目录，建立 {时间戳: 文件路径} 的字典
    避免在循环中使用 glob，彻底解决 pathlib 报错
    """
    print(f"📂 正在索引目录: {root_path} ...")
    index = {}
    # 使用 os.walk 替代 pathlib 的 glob，这是最稳健的递归方式
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.endswith(".tif") and suffix_key in f:
                # 提取文件名中的时间戳，假设格式为 20230105_12
                # 这里根据你的文件名规则微调
                import re
                match = re.search(r'(\d{8}_\d{2})', f)
                if match:
                    index[match.group(1)] = os.path.join(root, f)
    print(f"✅ 索引完成，共找到 {len(index)} 个有效文件。")
    return index


def read_raster_and_coords(tif_path):
    with rasterio.open(tif_path) as src:
        # 直接读取数据，不进行手动 flipud
        data = src.read(1).astype(np.float32)
        data[data == src.nodata] = np.nan

        # 获取地理转换参数
        bounds = src.bounds
        # 生成经纬度序列
        # 注意：这里要确保生成的维度与 data 的 shape 完全一致
        lons = np.linspace(bounds.left, bounds.right, src.width)
        # 关键点：对于 TIF 这种 Top-Down 数据，纬度应该从高到低排列
        lats = np.linspace(bounds.top, bounds.bottom, src.height)

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        return data, lon_grid, lat_grid

def plot_case_matrix_real_wind(case_row, clipped_prov, clipped_hz, pm_index, u_index, v_index):
    case_name = case_row['冬季案例']
    start_t = pd.to_datetime(case_row['分析开始'])
    end_t = pd.to_datetime(case_row['分析结束'])

    target_times = [start_t + timedelta(seconds=s) for s in np.linspace(0, (end_t - start_t).total_seconds(), 6)]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13), constrained_layout=True)
    axes = axes.flatten()

    im = None
    q = None

    for i, t in enumerate(target_times):
        ax = axes[i]
        t_key = t.strftime('%Y%m%d_%H')

        pm_path = pm_index.get(t_key)
        u_path = u_index.get(t_key)
        v_path = v_index.get(t_key)

        # 1. 绘制 PM2.5 浓度场
        if pm_path:
            with rasterio.open(pm_path) as src:
                pm_data = src.read(1).astype(np.float32)
                pm_data[(pm_data < 0) | (pm_data > 800)] = np.nan
                im = ax.imshow(pm_data, cmap='jet', vmin=0, vmax=150,
                               extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], origin='upper')

        # 2. 绘制风场 (加入调试打印)
        if u_path and v_path:
            u_data, lon_g, lat_g = read_raster_and_coords(u_path)
            v_data, _, _ = read_raster_and_coords(v_path)

            # --- 诊断打印（只打印第一个子图） ---
            if i == 0:
                print(f"📊 风场数值诊断: U范围({np.nanmin(u_data):.2f} to {np.nanmax(u_data):.2f}), "
                      f"V范围({np.nanmin(v_data):.2f} to {np.nanmax(v_data):.2f})")

            # 过滤掉包含 NaN 的点，否则 quiver 可能会报错
            mask = ~np.isnan(u_data) & ~np.isnan(v_data)

            # # 抽样间隔：5km 数据在长三角范围内，skip=8 到 10 比较合适
            # skip = 8
            #
            # # 【核心修改】Quiver 参数：
            # # scale=None 让 Matplotlib 自动计算合适的比例
            # # 如果自动计算还是看不见，手动尝试 scale=30 或 50
            # q = ax.quiver(lon_g[mask][::skip], lat_g[mask][::skip],
            #               u_data[mask][::skip], v_data[mask][::skip],
            #               color='white',  # 白色在深色底图上更明显
            #               scale=50,  # 减小 scale 会让箭头变长
            #               width=0.003,  # 稍微加粗
            #               alpha=0.8, zorder=15)
            # --- 物理逻辑检查：冬季西北风 ---
            # 如果你发现箭头还是指向西北，说明 V 分量需要手动修正符号
            # 在某些 ERA5 导出过程中，y 轴的方向定义可能导致 V 符号相反
            v_corrected = v_data  # 尝试翻转 V 分量的符号

            skip = 5
            q = ax.quiver(lon_g[::skip, ::skip], lat_g[::skip, ::skip],
                          u_data[::skip, ::skip], v_corrected[::skip, ::skip],
                          color='white',
                          scale=50,
                          width=0.004,
                          zorder=15)

        # 3. 边界叠加
        if not clipped_prov.empty:
            clipped_prov.plot(ax=ax, linewidth=0.6, edgecolor='#333333', facecolor='none', alpha=0.5, zorder=2)
        clipped_hz.plot(ax=ax, linewidth=1.8, edgecolor='darkblue', facecolor='none', zorder=10)

        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_title(f"{t.strftime('%Y-%m-%d %H:00')}", fontsize=14, fontweight='bold')
        ax.set_axis_off()

    # 添加图例
    if im:
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.7, aspect=20, pad=0.02)
        cbar.set_label('PM2.5 浓度 ($\mu g/m^3$)', fontsize=14)
        if q:
            # 比例尺箭头，10m/s 的参考长度
            axes[-1].quiverkey(q, 0.9, 0.05, 10, r'$10 \ m/s$', labelpos='E', coordinates='figure', color='black')

    save_path = OUT_DIR / f"Evolution_RealWind_{case_name.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 案例渲染完成: {case_name}")

# ==============================
# --- 2. 修改后的 Main 函数 (检查索引结果) ---
# ==============================
def main():
    # 建立索引
    pm_index = build_file_index(PM25_ROOT)
    u_index = build_file_index(U10_ROOT, "u10")
    v_index = build_file_index(V10_ROOT, "v10")

    # --- 诊断打印：确认索引里到底有没有东西 ---
    print(f"📊 索引统计:")
    print(f"   - PM2.5 索引条目数: {len(pm_index)}")
    print(f"   - U10 索引条目数: {len(u_index)}")
    print(f"   - V10 索引条目数: {len(v_index)}")

    if len(u_index) == 0:
        print("❌ 警告：U10 索引为空！请检查 build_file_index 里的正则表达式或文件夹路径。")

    # 准备 SHP
    research_bbox = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    try:
        prov = gpd.read_file(PROVINCE_SHP_PATH, encoding='utf-8')
        clipped_prov = gpd.clip(prov, research_bbox)
        cities = gpd.read_file(CITY_SHP_PATH, encoding='utf-8')
        clipped_hz = gpd.clip(cities[cities['name'].str.contains('杭州')], research_bbox)
    except Exception as e:
        print(f"❌ SHP 加载失败: {e}")
        return

    # 绘图
    cases_df = pd.read_csv(CASES_CSV)
    for _, row in cases_df.iterrows():
        print(f"🎬 正在渲染案例：{row['冬季案例']} ...")
        plot_case_matrix_real_wind(row, clipped_prov, clipped_hz, pm_index, u_index, v_index)

    print(f"✨ 全部完成！结果已存入：{OUT_DIR}")
if __name__ == "__main__":
    main()