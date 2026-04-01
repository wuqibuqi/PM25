import os, glob, joblib, gc, re, warnings
import numpy as np
import pandas as pd
import rasterio
import torch
from tqdm import tqdm
from datetime import datetime

# 彻底屏蔽无用警告
warnings.filterwarnings("ignore", category=UserWarning)

# ================= 1. 路径与特征配置 =================
MODEL_DIR = r"E:\01Output\Experiments_new\Model_Output\RF_552\Train_Data_Exp2_Time"
MODEL_A_PATH = os.path.join(MODEL_DIR, "Train_Data_Exp2_Time_rf_AOD_model.pkl")
MODEL_B_PATH = os.path.join(MODEL_DIR, "Train_Data_Exp2_Time_rf_Meteo_model.pkl")

STATION_CSV = r"E:\01Output\Experiments_new\Train_Data_Exp2_Time.csv"
DATA_ROOT = r"E:\Standard_Dataset_5km"
AOD_ROOT = r"E:\Himawari-8_TIFF_24h"
OUTPUT_DIR = r"E:\PM25_Retrieval_Results_Full_Res"  # 建议换个新文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 特征顺序必须与训练时严格一致
features_A = [
    "Longitude", "Latitude", "DOY", "hour", "DEM_5km", "Pop_5km", "Roads_5km",
    "CLCD_1_Cropland_Fraction_5km", "CLCD_2_Forest_Fraction_5km", "CLCD_5_Water_Fraction_5km",
    "CLCD_8_Impervious_Fraction_5km", "blh", "rh", "t2m", "sp", "u10", "v10", "NDVI", "AOD"
]
features_B = [f for f in features_A if f != "AOD"]
era5_vars = ['blh', 'rh', 't2m', 'sp', 'u10', 'v10']


# ================= 2. 核心加速工具 =================

def build_fast_index(root_path):
    """预扫描建立文件索引字典，解决 4 万文件带来的 I/O 延迟"""
    print(f"📂 正在扫描目录索引: {root_path} ...")
    idx = {}
    for r, d, files in os.walk(root_path):
        for f in files:
            if f.endswith(".tif"):
                match = re.search(r'(\d{8}_\d{2})', f)
                if match:
                    key = match.group(1)
                    var_name = "AOD"
                    for v in era5_vars:
                        if f"_{v}" in f or f"{v}_" in f:
                            var_name = v
                            break
                    idx[f"{key}_{var_name}"] = os.path.join(r, f)
    return idx


# ================= 3. 【全分辨率核心】GPU 分块 IDW 引擎 =================

def gpu_idw_full_res_engine(st_coords, res, grid_coords_full, device, p=2):
    """
    全分辨率 IDW：不降采样，利用 GPU 分块计算 35 万像素点的残差
    """
    if len(res) == 0:
        return np.zeros(grid_coords_full.shape[0], dtype=np.float32)

    num_pixels = grid_coords_full.shape[0]
    final_res = torch.zeros(num_pixels, device=device, dtype=torch.float32)

    # 将数据搬运到 GPU
    st_xyz = torch.tensor(st_coords, device=device, dtype=torch.float32)
    res_tensor = torch.tensor(res, device=device, dtype=torch.float32)
    grid_coords_torch = torch.tensor(grid_coords_full, device=device, dtype=torch.float32)

    # 【分块策略】针对 3060 6G 显存，每块处理 40,000 个像素点
    chunk_size = 40000

    with torch.no_grad():
        for i in range(0, num_pixels, chunk_size):
            end = min(i + chunk_size, num_pixels)
            grid_chunk = grid_coords_torch[i:end]

            # 计算欧式距离矩阵 [chunk_size, num_stations]
            dist = torch.cdist(grid_chunk, st_xyz)

            # IDW 权重计算: 1/d^p
            weight = 1.0 / (dist ** p + 1e-6)
            sum_w = torch.sum(weight, dim=1)

            # 算出该块的残差插值并存入结果
            final_res[i:end] = torch.sum(weight * res_tensor, dim=1) / sum_w

            # 及时清理临时变量，释放显存
            del dist, weight, sum_w

    return final_res.cpu().numpy()


# ================= 4. 主反演流程 =================

def run_retrieval_full_res():
    print("🧠 正在加载模型 (Memory Mapping Mode)...")
    # 共享内存模式，防止 5800H 的 RAM 爆掉
    rf_A = joblib.load(MODEL_A_PATH, mmap_mode='r')
    rf_B = joblib.load(MODEL_B_PATH, mmap_mode='r')

    # 显卡环境初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"✅ 已成功连接显卡: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 严重警告：未检测到 GPU！全分辨率计算将极其缓慢，请检查 PyTorch CUDA 环境！")

    # 地面站实测值规范化 (适配 PM25_5030)
    df_st = pd.read_csv(STATION_CSV)
    if 'PM25_5030' in df_st.columns:
        df_st.rename(columns={'PM25_5030': 'PM25'}, inplace=True)
    df_st['UTC_Time'] = pd.to_datetime(df_st['UTC_Time'])

    # 建立索引
    era5_index = build_fast_index(os.path.join(DATA_ROOT, "ERA5"))
    aod_index = build_fast_index(AOD_ROOT)

    print("🗺️ 预载空间网格与静态图层...")
    with rasterio.open(os.path.join(DATA_ROOT, "Static", "Roads_5km.tif")) as src:
        meta, height, width = src.meta.copy(), src.height, src.width
        xs, ys = rasterio.transform.xy(src.transform, *np.meshgrid(np.arange(width), np.arange(height)))
        grid_coords_full = np.column_stack((np.array(xs).flatten(), np.array(ys).flatten()))

    # 缓存静态矩阵
    static_feats = ["DEM_5km", "Pop_5km", "Roads_5km", "CLCD_1_Cropland_Fraction_5km",
                    "CLCD_2_Forest_Fraction_5km", "CLCD_5_Water_Fraction_5km", "CLCD_8_Impervious_Fraction_5km"]
    static_arrays = {f: rasterio.open(os.path.join(DATA_ROOT, "Static", f"{f}.tif")).read(1).flatten() for f in
                     static_feats}

    utc_times = pd.date_range(start="2020-01-01", end="2024-12-31 23:00", freq="h")

    for utc in tqdm(utc_times, desc="🔥 全分辨率反演"):
        t_str = utc.strftime('%Y%m%d_%H')
        out_dir = os.path.join(OUTPUT_DIR, utc.strftime('%Y/%m/%d'))
        out_path = os.path.join(out_dir, f"PM25_FullRes_{t_str}_UTC.tif")
        if os.path.exists(out_path): continue

        # --- 特征矩阵 X 构造 (NumPy 向量化) ---
        X = np.zeros((grid_coords_full.shape[0], len(features_A)), dtype=np.float32)
        X[:, 0], X[:, 1] = grid_coords_full[:, 0], grid_coords_full[:, 1]
        X[:, 2], X[:, 3] = utc.dayofyear, utc.hour
        for i, f in enumerate(static_feats): X[:, 4 + i] = static_arrays[f]

        try:
            with rasterio.open(os.path.join(DATA_ROOT, "NDVI_Daily", f"NDVI_{utc.strftime('%Y%m%d')}.tif")) as s:
                X[:, 17] = s.read(1).flatten()
            for i, v in enumerate(era5_vars):
                with rasterio.open(era5_index[f"{t_str}_{v}"]) as s: X[:, 11 + i] = s.read(1).flatten()
        except:
            continue

        aod_p = aod_index.get(f"{t_str}_AOD")
        if aod_p:
            with rasterio.open(aod_p) as s:
                a = s.read(1).flatten()
                X[:, 18] = np.where((a > 0) & (a < 5), a, np.nan)
        else:
            X[:, 18] = np.nan

        # --- 第一步：RF 预测 ---
        final_pm = np.full(grid_coords_full.shape[0], np.nan, dtype=np.float32)
        mask_a = ~np.isnan(X[:, 18])
        if mask_a.any(): final_pm[mask_a] = rf_A.predict(X[mask_a])
        mask_b = np.isnan(X[:, 18])
        if mask_b.any(): final_pm[mask_b] = rf_B.predict(X[mask_b, :-1])

        # --- 第二步：全分辨率 GPU 残差修正 ---
        df_h = df_st[df_st['UTC_Time'] == utc].copy()
        if len(df_h) >= 3:
            df_h['DOY'], df_h['hour'] = utc.dayofyear, utc.hour
            st_X = df_h[features_A].values
            st_preds = np.zeros(len(df_h))
            m_a = ~np.isnan(st_X[:, -1])
            if m_a.any(): st_preds[m_a] = rf_A.predict(st_X[m_a])
            if (~m_a).any(): st_preds[~m_a] = rf_B.predict(st_X[~m_a, :-1])

            res = df_h['PM25'].values - st_preds
            # 调用全分辨率 GPU 引擎 (不进行降采样)
            correction = gpu_idw_full_res_engine(df_h[['Longitude', 'Latitude']].values, res,
                                                 grid_coords_full, device)
            final_pm = np.where(np.isnan(final_pm), np.nan, final_pm + correction)

        # --- 保存结果 ---
        os.makedirs(out_dir, exist_ok=True)
        out_img = np.clip(final_pm.reshape(height, width), 0, 1000).astype(np.float32)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(out_img, 1)

        del X, final_pm;
        gc.collect()


if __name__ == "__main__":
    run_retrieval_full_res()