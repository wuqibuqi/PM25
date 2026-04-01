import os
import glob
import joblib
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from datetime import datetime
import gc

# ================= 1. 路径与核心配置 =================
# 根目录 (D盘固态)
BASE_DATA_DIR = r"D:\1document\Graduation Thesis\01Code\DATA"

# 模型路径
MODEL_BASE_PATH = os.path.join(BASE_DATA_DIR, "Final_Paper_Result", "Paper_Base_Model.pkl")
MODEL_RES_PATH = os.path.join(BASE_DATA_DIR, "Final_Paper_Result", "Paper_Res_Model.pkl")

# 数据源路径
DATA_ROOT = os.path.join(BASE_DATA_DIR, "Standard_Dataset_5km")
AOD_ROOT = os.path.join(BASE_DATA_DIR, "Himawari-8_TIFF_24h")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "PM25_Retrieval_Results_Seamless")

# --- 反演控制开关 ---
OVERWRITE = False  # True: 覆盖已有文件 | False: 断点续传 (跳过已存在的文件)

# 变量定义
era5_vars = ['blh', 'rh', 't2m', 'sp', 'u10', 'v10']
static_feats = ["DEM_5km", "Pop_5km", "Roads_5km",
                "CLCD_1_Cropland_Fraction_5km", "CLCD_2_Forest_Fraction_5km",
                "CLCD_5_Water_Fraction_5km", "CLCD_8_Impervious_Fraction_5km"]


# ================= 2. 核心反演引擎 =================
def run_inversion_engine(start_date, end_date):
    print(f"🧠 正在载入两阶段模型... (模式: {'覆盖' if OVERWRITE else '续传'})")
    model_base = joblib.load(MODEL_BASE_PATH)
    model_res = joblib.load(MODEL_RES_PATH)

    print("🗺️ 初始化地理基准与静态特征...")
    base_path = os.path.join(DATA_ROOT, "Static", "Roads_5km.tif")
    with rasterio.open(base_path) as src:
        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32, nodata=np.nan)
        height, width = src.height, src.width
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        base_lons = np.array(xs).flatten()
        base_lats = np.array(ys).flatten()

    num_pixels = len(base_lons)

    # 预载静态特征 (常驻内存)
    static_matrix = np.zeros((num_pixels, 7), dtype=np.float32)
    for i, feat in enumerate(static_feats):
        p = os.path.join(DATA_ROOT, "Static", f"{feat}.tif")
        with rasterio.open(p) as s:
            static_matrix[:, i] = s.read(1).flatten()

    days = pd.date_range(start=start_date, end=end_date, freq="D")

    for day in tqdm(days, desc="📅 总进度"):
        date_str = day.strftime('%Y%m%d')
        yyyy, mm, dd = day.strftime('%Y'), day.strftime('%m'), day.strftime('%d')

        # 目标文件夹结构: OUTPUT/Year/Month/Day
        day_out_dir = os.path.join(OUTPUT_DIR, yyyy, mm, dd)

        # --- 断点续传逻辑 ---
        if not OVERWRITE:
            # 如果全天 24 个文件都存在，则跳过这一天
            if len(glob.glob(os.path.join(day_out_dir, "*.tif"))) == 24:
                continue

        os.makedirs(day_out_dir, exist_ok=True)

        # 1. 每天只读一次 NDVI (提速关键)
        ndvi_p = os.path.join(DATA_ROOT, "NDVI_Daily", f"NDVI_{date_str}.tif")
        if not os.path.exists(ndvi_p): continue
        with rasterio.open(ndvi_p) as s:
            day_ndvi = s.read(1).flatten()

        # 2. 准备 24 小时的大堆叠矩阵
        # 基础特征 (N*24, 18) | 残差补充特征 (N*24, 2)
        X_day_base = np.zeros((num_pixels * 24, 18), dtype=np.float32)
        day_aod_extra = np.zeros((num_pixels * 24, 2), dtype=np.float32)  # [Value, Flag]

        processed_hours = []
        for h in range(24):
            hh_str = f"{h:02d}"
            out_name = f"PM25_Seamless_{date_str}_{hh_str}00.tif"
            out_path = os.path.join(day_out_dir, out_name)

            if not OVERWRITE and os.path.exists(out_path):
                continue

            try:
                # 读取该小时 ERA5
                hour_meteo = np.zeros((num_pixels, 6), dtype=np.float32)
                meteo_valid = True
                for i, var in enumerate(era5_vars):
                    p = os.path.join(DATA_ROOT, "ERA5", var, yyyy, mm, f"ERA5_{date_str}_{hh_str}00_{var}.tif")
                    if not os.path.exists(p): meteo_valid = False; break
                    with rasterio.open(p) as s:
                        hour_meteo[:, i] = s.read(1).flatten()

                if not meteo_valid: continue

                start, end = h * num_pixels, (h + 1) * num_pixels
                # 填充基础矩阵块
                X_day_base[start:end, 0] = base_lons
                X_day_base[start:end, 1] = base_lats
                X_day_base[start:end, 2] = day.dayofyear
                X_day_base[start:end, 3] = h
                X_day_base[start:end, 4:11] = static_matrix
                X_day_base[start:end, 11:17] = hour_meteo
                X_day_base[start:end, 17] = day_ndvi

                # 处理 AOD 路径及读取
                aod_sub_dir = os.path.join(AOD_ROOT, f"{yyyy}{mm}", dd)
                aod_f_list = glob.glob(os.path.join(aod_sub_dir, f"*{date_str}_{hh_str}00_*.tif"))

                if aod_f_list:
                    with rasterio.open(aod_f_list[0]) as s:
                        aod_raw = s.read(1).flatten()
                        mask = (~np.isnan(aod_raw)) & (aod_raw > 0)
                        day_aod_extra[start:end, 1][mask] = 1  # Flag
                        day_aod_extra[start:end, 0][mask] = aod_raw[mask]  # Value

                processed_hours.append(h)
            except Exception:
                continue

        if not processed_hours: continue

        # 3. 极速推理 (全天候大矩阵直接喂入模型)
        # 只选取有数据的小时进行预测
        max_h = max(processed_hours) + 1
        valid_slice = slice(0, max_h * num_pixels)

        # 第一阶段：背景场
        pm_base_day = model_base.predict(X_day_base[valid_slice])
        # 第二阶段：残差修正
        X_res_day = np.column_stack([X_day_base[valid_slice], day_aod_extra[valid_slice]])
        pm_res_day = model_res.predict(X_res_day)

        # 融合与物理截断
        final_pm_day = np.maximum(pm_base_day + pm_res_day, 0)

        # 4. 结果分块输出为 TIF
        for h in processed_hours:
            hh_str = f"{h:02d}"
            out_file = os.path.join(day_out_dir, f"PM25_Seamless_{date_str}_{hh_str}00.tif")

            h_data = final_pm_day[h * num_pixels: (h + 1) * num_pixels]
            out_img = h_data.reshape(height, width)

            with rasterio.open(out_file, 'w', **meta) as dst:
                dst.write(out_img.astype(np.float32), 1)

        # 每日结束手动释放内存
        del X_day_base, day_aod_extra, pm_base_day, pm_res_day, final_pm_day
        gc.collect()

    print(f"\n✨ 任务圆满完成！所有结果均存储于: {OUTPUT_DIR}")


if __name__ == "__main__":
    # 设定反演时间起止
    START = "2020-01-01"
    END = "2024-01-01"
    run_inversion_engine(START, END)