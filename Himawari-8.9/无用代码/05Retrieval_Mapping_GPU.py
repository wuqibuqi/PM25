# 必须放在所有 import 之前，锁定底层环境
import os

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
import gc
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor

# ================= 1. 路径配置 (请核对) =================
BASE_DATA_DIR = r"D:\1document\Graduation Thesis\01Code\DATA"
MODEL_DIR = os.path.join(BASE_DATA_DIR, "Final_Paper_Result")
DATA_ROOT = os.path.join(BASE_DATA_DIR, "Standard_Dataset_5km")
AOD_ROOT = os.path.join(BASE_DATA_DIR, "Himawari-8_TIFF_24h")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "PM25_Results_Extreme_GPU")


# ================= 2. 核心加速组件 =================

def load_gpu_model(model_name):
    """自动转换并加载最适合 GPU 推理的原生模型格式"""
    pkl_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    txt_path = os.path.join(MODEL_DIR, f"{model_name}.txt")
    if not os.path.exists(txt_path):
        print(f"🔄 转换模型: {model_name}...")
        m = joblib.load(pkl_path)
        booster = m.booster_ if hasattr(m, 'booster_') else m
        booster.save_model(txt_path)
    return lgb.Booster(model_file=txt_path)


def fast_read(path):
    """多线程专用的高速读取函数"""
    try:
        with rasterio.open(path) as s:
            return s.read(1).flatten()
    except:
        return None


def build_aod_index(root_dir):
    """预建索引，消除磁盘搜索耗时"""
    print("🔍 建立 AOD 索引中...")
    index = {}
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.tif'):
                # 提取关键时间戳，如 '20200101_0800'
                index[f] = os.path.join(root, f)
    return index


# ================= 3. 极速推理引擎 =================

def run_extreme_engine(start_date, end_date):
    # 初始化
    model_base = load_gpu_model("Paper_Base_Model")
    model_res = load_gpu_model("Paper_Res_Model")
    aod_index = build_aod_index(AOD_ROOT)

    # GPU 推理参数显式配置
    # 注意：如果 nvidia-smi 显示利用率为 0，说明你安装的 LightGBM 不是 GPU 版
    gpu_params = {
        'task': 'predict',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'num_threads': 4  # 配合 GPU 的辅助线程
    }

    # 预载静态数据
    print("🗺️ 载入静态要素...")
    static_feats = ["DEM_5km", "Pop_5km", "Roads_5km",
                    "CLCD_1_Cropland_Fraction_5km", "CLCD_2_Forest_Fraction_5km",
                    "CLCD_5_Water_Fraction_5km", "CLCD_8_Impervious_Fraction_5km"]

    with rasterio.open(os.path.join(DATA_ROOT, "Static", "Roads_5km.tif")) as src:
        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32, nodata=np.nan)
        h_dim, w_dim = src.height, src.width
        xs, ys = rasterio.transform.xy(src.transform, *np.meshgrid(np.arange(w_dim), np.arange(h_dim)))
        coords = np.column_stack([np.array(xs).flatten(), np.array(ys).flatten()])

    num_pixels = len(coords)
    static_data = np.stack([fast_read(os.path.join(DATA_ROOT, "Static", f"{f}.tif")) for f in static_feats], axis=1)

    # 时间序列处理
    dates = pd.date_range(start_date, end_date)
    for day in tqdm(dates, desc="🚀 极速反演中"):
        d_str = day.strftime('%Y%m%d')
        yyyy, mm, dd = day.strftime('%Y'), day.strftime('%m'), day.strftime('%d')

        day_dir = os.path.join(OUTPUT_DIR, yyyy, mm, dd)
        if os.path.exists(day_dir) and len(os.listdir(day_dir)) >= 24: continue

        # 准备单日容器
        X_day = np.zeros((num_pixels * 24, 18), dtype=np.float32)
        A_extra = np.zeros((num_pixels * 24, 2), dtype=np.float32)
        active_h = []

        # 获取 NDVI
        ndvi_p = os.path.join(DATA_ROOT, "NDVI_Daily", f"NDVI_{d_str}.tif")
        day_ndvi = fast_read(ndvi_p)
        if day_ndvi is None: continue

        # 每小时数据采集
        for h in range(24):
            hh = f"{h:02d}"
            # 气象文件列表
            met_paths = [os.path.join(DATA_ROOT, "ERA5", v, yyyy, mm, f"ERA5_{d_str}_{hh}00_{v}.tif")
                         for v in ['blh', 'rh', 't2m', 'sp', 'u10', 'v10']]

            # 使用 ThreadPoolExecutor 并行读取当小时所有文件（IO 提速核心）
            with ThreadPoolExecutor(max_workers=8) as executor:
                met_results = list(executor.map(fast_read, met_paths))

            if any(r is None for r in met_results): continue

            # 填充特征矩阵
            r_s, r_e = h * num_pixels, (h + 1) * num_pixels
            X_day[r_s:r_e, 0:2] = coords
            X_day[r_s:r_e, 2] = day.dayofyear
            X_day[r_s:r_e, 3] = h
            X_day[r_s:r_e, 4:11] = static_data
            X_day[r_s:r_e, 11:17] = np.stack(met_results, axis=1)
            X_day[r_s:r_e, 17] = day_ndvi

            # 匹配 AOD
            aod_key = f"{d_str}_{hh}00"  # 模糊匹配关键字符
            target = next((p for n, p in aod_index.items() if aod_key in n), None)
            if target:
                aod_v = fast_read(target)
                if aod_v is not None:
                    m = (~np.isnan(aod_v)) & (aod_v > 0)
                    A_extra[r_s:r_e, 0][m] = aod_v[m]
                    A_extra[r_s:r_e, 1][m] = 1

            active_h.append(h)

        if not active_h: continue

        # --- GPU 推理阶段 ---
        # 显式传递参数，强制引导 LightGBM 使用 GPU（如果环境支持）
        try:
            pm_base = model_base.predict(X_day, **gpu_params)
            pm_res = model_res.predict(np.column_stack([X_day, A_extra]), **gpu_params)
        except:
            # 万一 GPU 报错（环境未编译），自动回退全核 CPU
            pm_base = model_base.predict(X_day)
            pm_res = model_res.predict(np.column_stack([X_day, A_extra]))

        final_pm = np.maximum(pm_base + pm_res, 0).reshape(24, num_pixels)

        # 保存结果
        os.makedirs(day_dir, exist_ok=True)
        for h in active_h:
            with rasterio.open(os.path.join(day_dir, f"PM25_{d_str}_{h:02d}00.tif"), 'w', **meta) as dst:
                dst.write(final_pm[h].reshape(h_dim, w_dim).astype(np.float32), 1)

        # 每一天跑完手动清理一次
        del X_day, A_extra, final_pm
        gc.collect()


if __name__ == "__main__":
    run_extreme_engine("2020-01-01", "2024-12-31")