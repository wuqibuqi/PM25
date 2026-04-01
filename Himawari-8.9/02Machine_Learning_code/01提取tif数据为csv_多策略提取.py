#优化后提速的提取特征
import pandas as pd
import rasterio
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings
# 忽略 np.nanmean 遇到全 NaN 时发出的警告，保持控制台清爽
warnings.filterwarnings('ignore', r'Mean of empty slice')
# --- 1. 路径与配置 ---
STATION_CSV = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\Output\Ground_Value_2020_2023_More.csv"
DATA_ROOT = r"E:\Standard_Dataset_5km"
AOD_ROOT = r"E:\Himawari-8_TIFF"
EXP_DIR = r"E:\01Output\Experiments_new"
os.makedirs(EXP_DIR, exist_ok=True)

static_features = ["DEM_5km", "Pop_5km", "Roads_5km",
                   "CLCD_1_Cropland_Fraction_5km", "CLCD_2_Forest_Fraction_5km",
                   "CLCD_3_Shrub_Fraction_5km", "CLCD_4_Grassland_Fraction_5km",
                   "CLCD_5_Water_Fraction_5km", "CLCD_8_Impervious_Fraction_5km"]
era5_vars = ['blh', 'd2m', 'lcc', 'sp', 't2m', 'tcc', 'u10', 'v10', 'rh']

VALID_BJT_HOURS = list(range(8, 18))


# ================= 提速核心武器：矩阵读取函数 =================
def fast_extract_points(tif_path, lons, lats):
    """一次性将 TIF 读入内存，利用 NumPy 矩阵索引瞬间提取所有坐标的值"""
    if not os.path.exists(tif_path):
        return np.full(len(lons), np.nan)

    with rasterio.open(tif_path) as src:
        # 1. 一次性将整张图读入内存矩阵 (提速千万倍的关键)
        data = src.read(1)
        # 2. 批量将经纬度转换为行列号
        rows, cols = rasterio.transform.rowcol(src.transform, lons, lats)
        rows, cols = np.array(rows), np.array(cols)

        # 3. 边界安全检查
        valid_mask = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)

        results = np.full(len(lons), np.nan)
        if valid_mask.any():
            # 4. 矩阵切片，瞬间提取
            vals = data[rows[valid_mask], cols[valid_mask]]
            # 5. 清洗无效值
            vals = np.where(vals < 1e10, vals, np.nan)
            results[valid_mask] = vals

    return results


def fast_spatial_extract(tif_path, lons, lats):
    """处理 AOD 3x3 空间扩展的加速函数"""
    if not os.path.exists(tif_path):
        return np.full(len(lons), np.nan)

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        h, w = src.height, src.width
        rows, cols = rasterio.transform.rowcol(src.transform, lons, lats)

        results = []
        for r, c in zip(rows, cols):
            if 0 <= r < h and 0 <= c < w:
                val = data[r, c]
                if val < 1e10:
                    results.append(val)
                else:
                    # 中心无效，求 3x3 均值
                    r_start, r_end = max(0, r - 1), min(h, r + 2)
                    c_start, c_end = max(0, c - 1), min(w, c + 2)
                    window = data[r_start:r_end, c_start:c_end]
                    valid = window[window < 1e10]
                    results.append(np.mean(valid) if valid.size > 0 else np.nan)
            else:
                results.append(np.nan)
    return np.array(results)


def get_aod_path(utc_time):
    folder = os.path.join(AOD_ROOT, utc_time.strftime("%Y%m"), utc_time.strftime("%d"))
    if not os.path.exists(folder): return None
    pattern = os.path.join(folder, f"H0*_{utc_time.strftime('%Y%m%d_%H')}00*.tif")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def start_experiment():
    print("📖 正在加载数据...")
    df = pd.read_csv(STATION_CSV, dtype={'StationCode': str}, low_memory=False)
    df['RealTime'] = pd.to_datetime(df['RealTime'])
    df['UTC_Time'] = pd.to_datetime(df['UTC_Time'])

    df = df[df['RealTime'].dt.hour.isin(VALID_BJT_HOURS)].copy()
    df = df[(df['RealTime'] >= '2020-01-01') & (df['RealTime'] <= '2023-12-31')].copy()

    print(f"📊 测试模式：共 {len(df)} 条样本。开始极速提取...")

    # ================= 优化点 1：静态特征降维打击 =================
    print("\n🏔️ 提取静态地理特征 (优化版: 只计算独立站点)...")
    # 提取唯一的站点列表 (大概只有2000行)
    unique_stations = df[['StationCode', 'Longitude', 'Latitude']].drop_duplicates().reset_index(drop=True)
    lons = unique_stations['Longitude'].values
    lats = unique_stations['Latitude'].values

    for feat in tqdm(static_features, desc="静态特征"):
        path = os.path.join(DATA_ROOT, "Static", f"{feat}.tif")
        unique_stations[feat] = fast_extract_points(path, lons, lats)

    # 通过 Merge 瞬间将静态特征分发给 552 万行
    df = pd.merge(df, unique_stations, on=['StationCode', 'Longitude', 'Latitude'], how='left')

    # ================= 优化点 2：动态特征矩阵提取 =================
    results = []
    # 以按小时打包的方式处理，每小时提取一次图
    for bjt_time, group in tqdm(df.groupby('RealTime'), desc="🚀 动态气象与多策略 AOD 提取"):
        h_data = group.copy()
        current_utc = h_data['UTC_Time'].iloc[0]

        # 将经纬度转为 NumPy 数组，极速运算
        lons = h_data['Longitude'].values
        lats = h_data['Latitude'].values

        yyyy_u, mm_u, dd_u, hh_u = current_utc.strftime("%Y"), current_utc.strftime("%m"), \
            current_utc.strftime("%d"), current_utc.strftime("%H")
        date_str_u = current_utc.strftime("%Y%m%d")

        # ERA5 极速提取
        for var in era5_vars:
            p = os.path.join(DATA_ROOT, "ERA5", var, yyyy_u, mm_u, f"ERA5_{date_str_u}_{hh_u}00_{var}.tif")
            h_data[var] = fast_extract_points(p, lons, lats)

        # NDVI 极速提取
        ndvi_p = os.path.join(DATA_ROOT, "NDVI_Daily", f"NDVI_{bjt_time.strftime('%Y%m%d')}.tif")
        h_data['NDVI'] = fast_extract_points(ndvi_p, lons, lats)

        # AOD 多路径准备
        p_t0 = get_aod_path(current_utc)
        p_tm1 = get_aod_path(current_utc - pd.Timedelta(hours=1))
        p_tp1 = get_aod_path(current_utc + pd.Timedelta(hours=1))

        # 向量化提取 T时刻, T-1, T+1 的 AOD 单点值
        v_t0 = fast_extract_points(p_t0, lons, lats) if p_t0 else np.full(len(h_data), np.nan)
        v_tm1 = fast_extract_points(p_tm1, lons, lats) if p_tm1 else np.full(len(h_data), np.nan)
        v_tp1 = fast_extract_points(p_tp1, lons, lats) if p_tp1 else np.full(len(h_data), np.nan)

        # 向量化提取 3x3 空间增强值
        v_space = fast_spatial_extract(p_t0, lons, lats) if p_t0 else np.full(len(h_data), np.nan)

        # 向量化计算策略 2: 宽松时间策略 (Exp2_Time)
        m2_time = np.copy(v_t0)
        mask_nan_t0 = np.isnan(v_t0)

        # 提取前后时刻的有效均值
        with np.errstate(invalid='ignore'):  # 忽略全部为 NaN 时的均值警告
            time_stack = np.vstack([v_tm1, v_tp1])
            time_mean = np.nanmean(time_stack, axis=0)

        m2_time[mask_nan_t0] = time_mean[mask_nan_t0]

        # 组合策略赋值
        h_data['Exp2_Time'] = m2_time
        h_data['Exp3_Space'] = v_space

        # np.where 实现极速的 if/else 优先选择
        h_data['Mix_PrioTime'] = np.where(~np.isnan(m2_time), m2_time, v_space)
        h_data['Mix_PrioSpace'] = np.where(~np.isnan(v_space), v_space, m2_time)

        results.append(h_data)

    final_df = pd.concat(results)

    # --- 3. 差异化保存 ---
    strategy_map = {
        "Exp2_Time": "Exp2_Time",
        "Exp3_Space": "Exp3_Space",
        "Mix_PrioTime": "Mix_PrioTime",
        "Mix_PrioSpace": "Mix_PrioSpace"
    }

    print("\n💾 正在保存四份实验总表...")
    for file_suffix, col_name in strategy_map.items():
        exp_df = final_df.copy()
        exp_df['AOD'] = exp_df[col_name]
        exp_df.drop(columns=['Exp2_Time', 'Exp3_Space', 'Mix_PrioTime', 'Mix_PrioSpace'], inplace=True)

        out_path = os.path.join(EXP_DIR, f"Train_Data_{file_suffix}.csv")
        exp_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"✅ 生成 {file_suffix}: AOD有效数 {exp_df['AOD'].notnull().sum()}")


if __name__ == "__main__":
    start_experiment()