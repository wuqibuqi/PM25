import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from shapely.geometry import box

# ================= 1. 路径与研究区配置 =================
input_csv = r"E:\01Output\Experiments_new\GTWR_Results_Monthly.csv"
save_dir = r"E:\01Output\Experiments_new\GTWR_Professional_Plots_Hourly"
os.makedirs(save_dir, exist_ok=True)

PROVINCE_SHP = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\中国_省\中国_省2.shp"
CITY_SHP = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\中国_市\中国_市2.shp"

# 研究区范围 [114-123E, 27-36N]
MIN_LON, MAX_LON = 114.0, 123.0
MIN_LAT, MAX_LAT = 27.0, 36.0
MAP_EXTENT = [MIN_LON, MAX_LON, MIN_LAT, MAX_LAT]

HZ_LAT, HZ_LON = 30.27, 120.15  # 杭州位置

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================= 2. 空间裁剪函数 =================
def get_clipped_features(extent):
    print(f"🌍 正在裁剪研究区底图...")
    bbox = box(extent[0] - 0.5, extent[2] - 0.5, extent[1] + 0.5, extent[3] + 0.5)

    gdf_prov = gpd.read_file(PROVINCE_SHP)
    clipped_prov = gpd.clip(gdf_prov, bbox)

    gdf_city = gpd.read_file(CITY_SHP)
    clipped_city = gpd.clip(gdf_city, bbox)

    prov_feat = cfeature.ShapelyFeature(clipped_prov.geometry, ccrs.PlateCarree(),
                                        edgecolor='#333333', facecolor='#FDFDFD', linewidth=1.0)
    city_feat = cfeature.ShapelyFeature(clipped_city.geometry, ccrs.PlateCarree(),
                                        edgecolor='#A0A0A0', facecolor='none', linewidth=0.5, alpha=0.6)
    return prov_feat, city_feat


# ================= 3. 增强版绘图引擎 =================
def plot_study_area_general(data, col_name, t_col=None, cmap='viridis', title="", vmin=None, vmax=None, save_name="",
                            label=""):
    """
    通用绘图函数
    t_col: 如果传入，则按 |t|>1.96 区分显著性；如果不传，则绘制所有点。
    """
    # 1. 过滤数据点
    mask = (data['Longitude'] >= MIN_LON) & (data['Longitude'] <= MAX_LON) & \
           (data['Latitude'] >= MIN_LAT) & (data['Latitude'] <= MAX_LAT)
    df_plot = data[mask].copy()

    # 2. 初始化画布
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_position([0.08, 0.08, 0.8, 0.8])
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    # 3. 添加底图
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#E3F2FD', zorder=0)
    ax.add_feature(prov_feat, zorder=1)
    ax.add_feature(city_feat, zorder=2)

    # 4. 确定显著性遮罩
    if t_col and t_col in df_plot.columns:
        sig_mask = np.abs(df_plot[t_col]) > 1.96
        # 绘制不显著点 (浅灰色)
        ax.scatter(df_plot.loc[~sig_mask, 'Longitude'], df_plot.loc[~sig_mask, 'Latitude'],
                   color='#D3D3D3', s=30, alpha=0.4, zorder=4, transform=ccrs.PlateCarree(), label='不显著 (p>0.05)')
    else:
        # 如果没有显著性列（如 RH），则所有点都视为“显著”显示
        sig_mask = pd.Series([True] * len(df_plot), index=df_plot.index)

    # 5. 绘制核心数据点
    sc = ax.scatter(df_plot.loc[sig_mask, 'Longitude'], df_plot.loc[sig_mask, 'Latitude'],
                    c=df_plot.loc[sig_mask, col_name],
                    cmap=cmap, s=120, alpha=0.9, edgecolors='k', linewidth=0.6,
                    vmin=vmin, vmax=vmax, zorder=5, transform=ccrs.PlateCarree())

    # 6. 标注受体点
    ax.plot(HZ_LON, HZ_LAT, marker='*', color='gold', markersize=20,
            markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10, label='受体点(杭州)')

    # 7. 网格线与刻度
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, color='gray', zorder=3)
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(MIN_LON, MAX_LON + 1, 2))
    gl.ylocator = mticker.FixedLocator(np.arange(MIN_LAT, MAX_LAT + 1, 2))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 8. 修饰
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.05)
    cbar.set_label(label, fontsize=12)
    ax.legend(loc='lower left', frameon=True, shadow=True, fontsize=10)

    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    plt.close()


# ================= 4. 执行流程 =================
print(f"📖 正在加载小时级数据: {os.path.basename(input_csv)}")
df = pd.read_csv(input_csv)

# 💡 核心修正：加入 numeric_only=True
spatial_df = df.groupby(['StationCode', 'Longitude', 'Latitude']).mean(numeric_only=True).reset_index()

# 预裁剪底图
prov_feat, city_feat = get_clipped_features(MAP_EXTENT)

# --- 任务 A: 局部 R2 ---
plot_study_area_general(
    spatial_df, 'Local_R2', t_col='t_value_AOD',
    cmap='YlGnBu', title="逐小时模型平均局部 $R^2$ 分布",
    vmin=0.6, vmax=0.9, save_name="StudyArea_Local_R2.png", label="Local $R^2$"
)

# --- 任务 B: AOD 影响系数 ---
study_sig = spatial_df[(np.abs(spatial_df['t_value_AOD']) > 1.96) &
                       (spatial_df['Longitude'].between(MIN_LON, MAX_LON))]
vmin_aod = np.percentile(study_sig['Coef_AOD'], 5)
vmax_aod = np.percentile(study_sig['Coef_AOD'], 95)

plot_study_area_general(
    spatial_df, 'Coef_AOD', t_col='t_value_AOD',
    cmap='RdYlBu_r', title="AOD 影响系数空间分异 (逐小时平均)",
    vmin=vmin_aod, vmax=vmax_aod,
    save_name="StudyArea_AOD_Coef.png", label="AOD 标准化回归系数"
)

# --- 任务 C: RH (相对湿度) 空间分布 ---
if 'rh' in spatial_df.columns:
    print("💧 正在绘制相对湿度空间分布图...")
    # 计算 RH 的显示范围（过滤掉可能的异常值）
    vmin_rh = np.percentile(spatial_df['rh'], 2)
    vmax_rh = np.percentile(spatial_df['rh'], 98)

    plot_study_area_general(
        spatial_df, 'rh', t_col=None,  # RH 不需要做显著性检验
        cmap='Blues', title="研究区平均相对湿度 ($RH$) 分布",
        vmin=vmin_rh, vmax=vmax_rh,
        save_name="StudyArea_RH_Mean.png", label="相对湿度 (%)"
    )

print(f"🎉 所有图件已生成！请查看文件夹：{save_dir}")