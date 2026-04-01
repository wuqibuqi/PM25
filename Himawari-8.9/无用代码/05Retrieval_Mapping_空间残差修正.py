import os, glob, joblib, gc, re
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# ================= 1. 路径与特征配置 =================
MODEL_DIR = r"E:\01Output\Experiments_new\Model_Output\RF_552\Train_Data_Exp2_Time"
MODEL_A_PATH = os.path.join(MODEL_DIR, "Train_Data_Exp2_Time_rf_AOD_model.pkl")
MODEL_B_PATH = os.path.join(MODEL_DIR, "Train_Data_Exp2_Time_rf_Meteo_model.pkl")

# 地面站实测数据路径
STATION_CSV = r"E:\01Output\Experiments_new\Train_Data_Exp2_Time.csv"

DATA_ROOT = r"E:\Standard_Dataset_5km"
AOD_ROOT = r"E:\Himawari-8_TIFF_24h"
OUTPUT_DIR = r"E:\PM25_Retrieval_Results_Corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 【关键】：特征顺序必须与模型训练时完全一致！
features_A = [
    "Longitude", "Latitude", "DOY", "hour", "DEM_5km", "Pop_5km", "Roads_5km",
    "CLCD_1_Cropland_Fraction_5km", "CLCD_2_Forest_Fraction_5km", "CLCD_5_Water_Fraction_5km",
    "CLCD_8_Impervious_Fraction_5km", "blh", "rh", "t2m", "sp", "u10", "v10", "NDVI", "AOD"
]
features_B = [f for f in features_A if f != "AOD"]
era5_vars = ['blh', 'rh', 't2m', 'sp', 'u10', 'v10']


# ================= 2. 预加载与加速工具 =================

def build_fast_index(root_path):
    """扫描全盘文件建立字典，消除循环内的 glob 耗时"""
    print(f"📂 正在快速扫描目录: {root_path} ...")
    idx = {}
    for r, d, files in os.walk(root_path):
        for f in files:
            if f.endswith(".tif"):
                match = re.search(r'(\d{8}_\d{2})', f)
                if match:
                    key = match.group(1)
                    # 识别变量类型
                    var_name = "AOD"
                    for v in era5_vars:
                        if f"_{v}" in f or f"{v}_" in f:
                            var_name = v
                            break
                    idx[f"{key}_{var_name}"] = os.path.join(r, f)
    return idx


def idw_smooth_correction(st_coords, res, grid_coords, k=12):
    """极速 IDW 残差修正算法"""
    tree = cKDTree(st_coords)
    # workers=-1 调用所有 CPU 核心计算距离
    dists, idx = tree.query(grid_coords, k=min(k, len(st_coords)), workers=-1)
    # 防止除零误差
    w = 1.0 / (dists ** 2 + 1e-6)
    return np.sum(w * res[idx], axis=1) / np.sum(w, axis=1)


# ================= 3. 核心主函数 =================

def run_fast_retrieval(start_date, end_date):
    print("🧠 正在初始化模型与资源...")
    rf_A = joblib.load(MODEL_A_PATH)
    rf_B = joblib.load(MODEL_B_PATH)

    # 1. 加载并规范化地面站数据
    df_st = pd.read_csv(STATION_CSV)
    # --- 【列名自动兼容逻辑】 ---
    possible_cols = ['PM25_5030', 'PM2.5', 'pm25', 'Value']
    for col in possible_cols:
        if col in df_st.columns:
            df_st.rename(columns={col: 'PM25'}, inplace=True)
            print(f"✅ 已识别实测值列名: {col} -> PM25")
            break

    df_st['UTC_Time'] = pd.to_datetime(df_st['UTC_Time'])

    # 2. 建立文件快查索引
    era5_index = build_fast_index(os.path.join(DATA_ROOT, "ERA5"))
    aod_index = build_fast_index(AOD_ROOT)

    # 3. 预载静态特征到内存，减少磁盘 I/O
    print("🏔️ 缓存静态特征至内存 (加速读取)...")
    static_feats = ["DEM_5km", "Pop_5km", "Roads_5km", "CLCD_1_Cropland_Fraction_5km",
                    "CLCD_2_Forest_Fraction_5km", "CLCD_5_Water_Fraction_5km", "CLCD_8_Impervious_Fraction_5km"]

    base_path = os.path.join(DATA_ROOT, "Static", "Roads_5km.tif")
    with rasterio.open(base_path) as src:
        meta, height, width = src.meta.copy(), src.height, src.width
        # 建立经纬度网格
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        grid_lons, grid_lats = np.array(xs).flatten(), np.array(ys).flatten()

    static_arrays = {f: rasterio.open(os.path.join(DATA_ROOT, "Static", f"{f}.tif")).read(1).flatten() for f in
                     static_feats}
    grid_coords = np.column_stack((grid_lons, grid_lats))

    # 4. 时间大循环
    utc_times = pd.date_range(start=start_date, end=f"{end_date} 23:00", freq="h")

    for utc in tqdm(utc_times, desc="🚀 极速反演中"):
        t_str = utc.strftime('%Y%m%d_%H')
        yyyy, mm, dd = utc.strftime('%Y'), utc.strftime('%m'), utc.strftime('%d')

        # 检查输出路径
        out_dir = os.path.join(OUTPUT_DIR, yyyy, mm, dd)
        out_path = os.path.join(out_dir, f"PM25_Corrected_{t_str}_UTC.tif")
        if os.path.exists(out_path): continue

        # --- 准备特征矩阵 ---
        X = np.zeros((len(grid_lons), len(features_A)), dtype=np.float32)
        X[:, 0], X[:, 1] = grid_lons, grid_lats
        X[:, 2], X[:, 3] = utc.dayofyear, utc.hour
        for i, f in enumerate(static_feats): X[:, 4 + i] = static_arrays[f]

        # 尝试填充动态特征，失败则跳过该小时
        try:
            # NDVI
            ndvi_p = os.path.join(DATA_ROOT, "NDVI_Daily", f"NDVI_{utc.strftime('%Y%m%d')}.tif")
            with rasterio.open(ndvi_p) as s:
                X[:, 17] = s.read(1).flatten()
            # ERA5
            for i, v in enumerate(era5_vars):
                with rasterio.open(era5_index[f"{t_str}_{v}"]) as s: X[:, 11 + i] = s.read(1).flatten()
        except:
            continue

        # AOD 特征
        aod_p = aod_index.get(f"{t_str}_AOD")
        if aod_p:
            with rasterio.open(aod_p) as s:
                aod = s.read(1).flatten()
                X[:, 18] = np.where((aod > 0) & (aod < 5), aod, np.nan)
        else:
            X[:, 18] = np.nan

        # --- 基础预测 ---
        final_pm = np.full(len(grid_lons), np.nan, dtype=np.float32)
        mask_a = ~np.isnan(X[:, 18])
        if mask_a.any(): final_pm[mask_a] = rf_A.predict(X[mask_a])
        mask_b = np.isnan(X[:, 18])
        if mask_b.any(): final_pm[mask_b] = rf_B.predict(X[mask_b, :-1])

        # --- 【核心新增】空间残差修正 ---
        df_h = df_st[df_st['UTC_Time'] == utc].copy()
        if len(df_h) >= 3:
            df_h['DOY'], df_h['hour'] = utc.dayofyear, utc.hour
            # 站点处的特征提取与预测
            st_X = df_h[features_A].values
            st_preds = np.zeros(len(df_h))
            # 逻辑分流：站点有 AOD 用模型 A，无 AOD 用模型 B
            m_st_a = ~np.isnan(st_X[:, -1])
            if m_st_a.any(): st_preds[m_st_a] = rf_A.predict(st_X[m_st_a])
            if (~m_st_a).any(): st_preds[~m_st_a] = rf_B.predict(st_X[~m_st_a, :-1])

            res = df_h['PM25'].values - st_preds
            # 将站点误差插值到全图，并叠加
            final_pm += idw_smooth_correction(df_h[['Longitude', 'Latitude']].values, res, grid_coords)

        # --- 写回 TIF 结果 ---
        os.makedirs(out_dir, exist_ok=True)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(np.clip(final_pm.reshape(height, width), 0, 800).astype(np.float32), 1)

        # 内存管理
        del X, final_pm;
        gc.collect()


if __name__ == "__main__":
    # 执行 5 年全量反演
    run_fast_retrieval("2020-01-01", "2024-12-31")