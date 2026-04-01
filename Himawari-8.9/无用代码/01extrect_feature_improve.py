#最开始的提取特征
import pandas as pd
import rasterio
import numpy as np
import os
import glob
from tqdm import tqdm

# --- 1. 路径与配置 ---
STATION_CSV = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\Output\Ground_Value_2020_2023.csv"
DATA_ROOT = r"E:\Standard_Dataset_5km"
AOD_ROOT = r"E:\Himawari-8_TIFF"
EXP_DIR = r"/Data/Output/Experiments"
os.makedirs(EXP_DIR, exist_ok=True)

static_features = ["DEM_5km", "Pop_5km", "Roads_5km",
                   "CLCD_1_Cropland_Fraction_5km", "CLCD_2_Forest_Fraction_5km",
                   "CLCD_3_Shrub_Fraction_5km", "CLCD_4_Grassland_Fraction_5km",
                   "CLCD_5_Water_Fraction_5km", "CLCD_8_Impervious_Fraction_5km"]
era5_vars = ['blh', 'd2m', 'lcc', 'sp', 't2m', 'tcc', 'u10', 'v10', 'rh']

# 允许的北京时间范围 (08:00 - 17:00 对应 UTC 0-9)
VALID_BJT_HOURS = list(range(8, 18))


def get_pixel_val(src, lon, lat, window=1):
    """提取像元值，解除 181 限制，完美复刻空间中心优先策略"""
    try:
        row, col = src.index(lon, lat)
        # 解除 181 限制，自动适应图片的实际尺寸 (src.height, src.width)
        if 0 <= row < src.height and 0 <= col < src.width:
            val = src.read(1)[row, col]
            is_valid_center = (val < 1e10)  # 判断中心点是否有效

            if window == 1:
                return val if is_valid_center else np.nan
            else:  # window == 3 (空间策略触发)
                # 规则1：如果该位置有数据，直接用这位置的数据
                if is_valid_center:
                    return val

                    # 规则2：如果没有，去找周围8个像素点，并算有效值的平均
                data = src.read(1)[max(0, row - 1):min(src.height, row + 2), max(0, col - 1):min(src.width, col + 2)]
                valid = data[data < 1e10]
                return np.mean(valid) if valid.size > 0 else np.nan
        return np.nan
    except:
        return np.nan


def get_aod_path(utc_time):
    """通过 UTC 时间匹配 AOD 文件"""
    folder = os.path.join(AOD_ROOT, utc_time.strftime("%Y%m"), utc_time.strftime("%d"))
    if not os.path.exists(folder): return None
    pattern = os.path.join(folder, f"H08_{utc_time.strftime('%Y%m%d_%H')}00*.tif")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def start_experiment():
    print("📖 正在加载数据并设置提取窗口...")
    df = pd.read_csv(STATION_CSV)
    df['RealTime'] = pd.to_datetime(df['RealTime'])
    df['UTC_Time'] = pd.to_datetime(df['UTC_Time'])

    # --- 1. 小时过滤：只留北京时间白天 (08:00 - 17:00) ---
    df = df[df['RealTime'].dt.hour.isin(VALID_BJT_HOURS)].copy()

    # --- 2. 日期过滤：处理全量数据 ---
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    df = df[(df['RealTime'] >= start_date) & (df['RealTime'] <= end_date)].copy()

    print(f"📊 测试模式：处理 {start_date} 到 {end_date}，共 {len(df)} 条样本")
    print(f"📊 预计循环次数 (小时包): {df['RealTime'].nunique()}")

    # 1. 静态特征提取
    print("🏔️ 提取静态地理特征...")
    for feat in static_features:
        path = os.path.join(DATA_ROOT, "Static", f"{feat}.tif")
        if os.path.exists(path):
            with rasterio.open(path) as src:
                df[feat] = df.apply(lambda r: get_pixel_val(src, r['Longitude'], r['Latitude']), axis=1)

    # 2. 动态特征与 AOD 多策略提取
    results = []
    # 按北京时间分组
    for bjt_time, group in tqdm(df.groupby('RealTime'), desc="时空多策略提取"):
        h_data = group.copy()
        current_utc = h_data['UTC_Time'].iloc[0]

        # ERA5 使用 UTC 时间
        yyyy_u, mm_u, dd_u, hh_u = current_utc.strftime("%Y"), current_utc.strftime("%m"), \
            current_utc.strftime("%d"), current_utc.strftime("%H")
        date_str_u = current_utc.strftime("%Y%m%d")

        # ERA5 提取
        for var in era5_vars:
            p = os.path.join(DATA_ROOT, "ERA5", var, yyyy_u, mm_u, f"ERA5_{date_str_u}_{hh_u}00_{var}.tif")
            if os.path.exists(p):
                with rasterio.open(p) as src:
                    h_data[var] = h_data.apply(lambda r: get_pixel_val(src, r['Longitude'], r['Latitude']), axis=1)
            else:
                h_data[var] = np.nan

        # NDVI 提取
        ndvi_p = os.path.join(DATA_ROOT, "NDVI_Daily", f"NDVI_{bjt_time.strftime('%Y%m%d')}.tif")
        if os.path.exists(ndvi_p):
            with rasterio.open(ndvi_p) as src:
                h_data['NDVI'] = h_data.apply(lambda r: get_pixel_val(src, r['Longitude'], r['Latitude']), axis=1)

        # AOD 多路径准备
        p_t0 = get_aod_path(current_utc)
        p_tm1 = get_aod_path(current_utc - pd.Timedelta(hours=1))
        p_tp1 = get_aod_path(current_utc + pd.Timedelta(hours=1))

        def calc_strategies(row):
            lon, lat = row['Longitude'], row['Latitude']

            # T时刻原始单点值
            v_t0 = np.nan
            if p_t0:
                with rasterio.open(p_t0) as s: v_t0 = get_pixel_val(s, lon, lat, 1)

            # T-1 和 T+1 时刻单点值
            v_tm1, v_tp1 = np.nan, np.nan
            if p_tm1:
                with rasterio.open(p_tm1) as s: v_tm1 = get_pixel_val(s, lon, lat, 1)
            if p_tp1:
                with rasterio.open(p_tp1) as s: v_tp1 = get_pixel_val(s, lon, lat, 1)

            # T时刻 3x3 空间值 (函数内已实现中心优先逻辑)
            v_space = np.nan
            if p_t0:
                with rasterio.open(p_t0) as s: v_space = get_pixel_val(s, lon, lat, 3)

            # --- 核心修改：完美复刻你的宽松时间策略 ---
            m2_time = v_t0
            if np.isnan(v_t0):  # 如果T时刻没有
                valid_t = []
                if not np.isnan(v_tm1): valid_t.append(v_tm1)  # T-1 有，加入备选
                if not np.isnan(v_tp1): valid_t.append(v_tp1)  # T+1 有，加入备选

                if len(valid_t) > 0:
                    m2_time = np.mean(valid_t)  # 如果只有一个就用它自己，如果有两个就算平均
                else:
                    m2_time = np.nan  # 前后都没有，那就真的没办法了

            # 策略 3: 空间增强
            m3_space = v_space

            # 策略 4: 优先时间 Mix_PrioTime
            m4_prio_time = m2_time if not np.isnan(m2_time) else m3_space

            # 策略 5: 优先空间 Mix_PrioSpace
            m5_prio_space = m3_space if not np.isnan(m3_space) else m2_time

            return pd.Series([m2_time, m3_space, m4_prio_time, m5_prio_space])

        h_data[['Exp2_Time', 'Exp3_Space', 'Mix_PrioTime', 'Mix_PrioSpace']] = h_data.apply(calc_strategies, axis=1)
        results.append(h_data)

    final_df = pd.concat(results)

    # --- 3. 差异化保存 ---
    strategy_map = {
        "Exp2_Time": "Exp2_Time",
        "Exp3_Space": "Exp3_Space",
        "Mix_PrioTime": "Mix_PrioTime",
        "Mix_PrioSpace": "Mix_PrioSpace"
    }

    print("\n💾 正在保存四份实验总表 (utf-8-sig)...")
    for file_suffix, col_name in strategy_map.items():
        exp_df = final_df.copy()
        exp_df['AOD'] = exp_df[col_name]
        exp_df.drop(columns=['Exp2_Time', 'Exp3_Space', 'Mix_PrioTime', 'Mix_PrioSpace'], inplace=True)

        out_path = os.path.join(EXP_DIR, f"Train_Data_{file_suffix}.csv")
        exp_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"✅ 生成 {file_suffix}: AOD有效数 {exp_df['AOD'].notnull().sum()}")


if __name__ == "__main__":
    start_experiment()