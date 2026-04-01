import pandas as pd
import rasterio
import numpy as np
import os
from tqdm import tqdm

# --- 1. 路径与配置 ---
STATION_CSV = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\Output\Ground_Value_2020_2023.csv"
DATA_ROOT = r"E:\Standard_Dataset_5km"
AOD_ROOT = r"E:\Himawari-8_TIFF"
OUTPUT_CSV = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\Output\RF_Train_Data_Final.csv"

# 静态特征
static_features = ["DEM_5km", "Pop_5km", "Roads_5km",
                   "CLCD_1_Cropland_Fraction_5km", "CLCD_2_Forest_Fraction_5km",
                   "CLCD_3_Shrub_Fraction_5km", "CLCD_4_Grassland_Fraction_5km",
                   "CLCD_5_Water_Fraction_5km", "CLCD_8_Impervious_Fraction_5km"]

# ERA5 变量
era5_vars = ['blh', 'd2m', 'lcc', 'sp', 't2m', 'tcc', 'u10', 'v10', 'rh']

# 【修正 1】：只保留北京时间白天的观测时段 (08:00 - 17:00)
VALID_BJT_HOURS = list(range(8, 18))


def get_val(src, lon, lat):
    """提取像元值"""
    try:
        row, col = src.index(lon, lat)
        # 【修正 2】：解除 181 限制，自动适应图片真实大小
        if 0 <= row < src.height and 0 <= col < src.width:
            val = src.read(1)[row, col]
            return val if val < 1e10 else np.nan
        return np.nan
    except:
        return np.nan


def start_extraction():
    print("📖 正在加载站点数据...")
    df = pd.read_csv(STATION_CSV)

    df['RealTime'] = pd.to_datetime(df['RealTime'])

    # 强制计算或转换 UTC 时间列 (RealTime 减去 8 小时)
    if 'UTC_Time' not in df.columns:
        df['UTC_Time'] = df['RealTime'] - pd.Timedelta(hours=8)
    else:
        df['UTC_Time'] = pd.to_datetime(df['UTC_Time'])

    # --- 核心优化 1：时段过滤 (只保留 BJT 08:00 - 17:00) ---
    initial_count = len(df)
    df = df[df['RealTime'].dt.hour.isin(VALID_BJT_HOURS)].copy()
    print(f"✂️ 过滤掉非观测时段数据: {initial_count} -> {len(df)} (保留了北京时间白天 10 小时样本)")

    # --- 核心优化 2：剔除降水 (如果有) ---
    if 'Precipitation' in df.columns:
        df.drop(columns=['Precipitation'], inplace=True)
        print("🗑️ 已剔除缺失严重的地面降水列。")

    # === 2. 提取静态特征 ===
    print("\n🏔️ 正在提取静态地理特征...")
    for feat in static_features:
        path = os.path.join(DATA_ROOT, "Static", f"{feat}.tif")
        if os.path.exists(path):
            with rasterio.open(path) as src:
                df[feat] = df.apply(lambda row: get_val(src, row['Longitude'], row['Latitude']), axis=1)

    # === 3. 提取动态特征 ===
    print(f"🛰️ 正在进行时空匹配提取 (目标范围: {len(df)} 样本)...")
    results = []

    # 按北京时间分组处理
    for bjt_time, group in tqdm(df.groupby('RealTime'), desc="逐小时匹配"):
        h_data = group.copy()

        # 【修正 3】：获取当前小时的 UTC 时间对象，用于拼凑文件名
        current_utc = bjt_time - pd.Timedelta(hours=8)

        yyyy_u = current_utc.strftime("%Y")
        mm_u = current_utc.strftime("%m")
        dd_u = current_utc.strftime("%d")
        hh_u = current_utc.strftime("%H")
        yyyy_mm_u = current_utc.strftime("%Y%m")
        date_str_u = current_utc.strftime("%Y%m%d")

        # NDVI 通常按北京时间的日期（Daily）来存储
        date_str_bjt = bjt_time.strftime("%Y%m%d")

        # --- A. 提取 AOD (葵花数据使用 UTC) ---
        aod_dir = os.path.join(AOD_ROOT, yyyy_mm_u, dd_u)
        aod_val = np.nan
        if os.path.exists(aod_dir):
            aod_files = [f for f in os.listdir(aod_dir) if date_str_u in f and f"_{hh_u}00_" in f]
            if aod_files:
                with rasterio.open(os.path.join(aod_dir, aod_files[0])) as src:
                    aod_val = h_data.apply(lambda row: get_val(src, row['Longitude'], row['Latitude']), axis=1)
        h_data['AOD'] = aod_val

        # --- B. 提取 ERA5 (使用 UTC) ---
        for var in era5_vars:
            era_path = os.path.join(DATA_ROOT, "ERA5", var, yyyy_u, mm_u, f"ERA5_{date_str_u}_{hh_u}00_{var}.tif")
            if os.path.exists(era_path):
                with rasterio.open(era_path) as src:
                    h_data[var] = h_data.apply(lambda row: get_val(src, row['Longitude'], row['Latitude']), axis=1)
            else:
                h_data[var] = np.nan

        # --- C. 提取 NDVI (使用 BJT) ---
        ndvi_path = os.path.join(DATA_ROOT, "NDVI_Daily", f"NDVI_{date_str_bjt}.tif")
        if os.path.exists(ndvi_path):
            with rasterio.open(ndvi_path) as src:
                h_data['NDVI'] = h_data.apply(lambda row: get_val(src, row['Longitude'], row['Latitude']), axis=1)

        results.append(h_data)

    final_df = pd.concat(results)

    # === 4. 智能清理与诊断 ===
    print("\n🔍 数据质量诊断 (空值数量):")
    print(final_df.isnull().sum()[final_df.isnull().sum() > 0])

    # 我们只要求【核心气象】和【静态特征】必须存在。AOD 缺失的行予以保留。
    essential_cols = ['t2m', 'blh', 'u10', 'DEM_5km', 'Pop_5km']

    before_drop = len(final_df)
    final_df.dropna(subset=essential_cols, inplace=True)

    print(f"\n✅ 核心匹配完成！")
    print(f"📊 原始时段样本: {before_drop}")
    print(f"📊 清理后样本量: {len(final_df)} (已保住 ERA5 匹配成功的行)")
    print(
        f"📡 其中 AOD 有效样本: {final_df['AOD'].notnull().sum()} ({final_df['AOD'].notnull().sum() / len(final_df):.1%})")

    if len(final_df) > 0:
        final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"🎉 特征大表已生成: {OUTPUT_CSV}")
    else:
        print("❌ 错误：清理后样本依然为 0，请检查 ERA5 重采样路径是否真的存在文件。")


if __name__ == "__main__":
    start_extraction()