import os
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from tqdm import tqdm
import shutil

# --- 1. 自动寻找模板 ---
AOD_ROOT = r"E:\Himawari-8_TIFF"


def get_master_tif(root_dir):
    """自动遍历文件夹，找到第一个 .tif 文件作为模板"""
    print(f"🔎 正在 {root_dir} 中自动寻找模板文件...")
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".tif"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"❌ 在 {root_dir} 下找不到任何 .tif 文件！请检查硬盘。")


MASTER_TIF = get_master_tif(AOD_ROOT)
print(f"🎯 成功锁定模板文件: {MASTER_TIF}")

# --- 2. 核心路径配置 ---
OUTPUT_ROOT = r"E:\Standard_Dataset_5km"

PATHS = {
    "STATIC": {
        #"CLCD": r"E:\CLCD_Data\CLCD_WGS84\CLCD_ROI_2020.tif",
        "DEM": r"E:\DEM\ChangSanJiao_DEM_30m.tif",
        "Pop": r"E:\Static_Features\Population\ROI_Pop_2020_100m.tif",
        "Roads": r"E:\Static_Features\Roads\Road_Density_5km_Forced_181.tif"
    },
    "DAILY": {
        "NDVI": r"E:\Static_Features\NDVI_Daily"
    },
    "HOURLY": {
        "ERA5": r"E:\ERA5_TIF"
    }
}


# def process_single_file(in_path, out_path, master_obj, resample_type):
#     """底层重采样/复制函数（强制覆盖）"""
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     if resample_type == 'copy':
#         shutil.copy(in_path, out_path)
#     else:
#         try:
#             data = rioxarray.open_rasterio(in_path)
#             method = Resampling.nearest if resample_type == 'nearest' else Resampling.bilinear
#             aligned = data.rio.reproject_match(master_obj, resampling=method)
#             # 保存，直接覆写
#             aligned.astype("float32").rio.to_raster(out_path, compress="lzw")
#         except Exception as e:
#             tqdm.write(f"❌ 错误: {in_path} -> {e}")

def process_single_file(in_path, out_path, master_obj, resample_type):
    """增加判断，如果文件存在则跳过"""
    # 如果输出文件已经存在，直接返回，不浪费计算资源
    if os.path.exists(out_path):
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if resample_type == 'copy':
        shutil.copy(in_path, out_path)
    else:
        try:
            data = rioxarray.open_rasterio(in_path)
            method = Resampling.nearest if resample_type == 'nearest' else Resampling.bilinear
            aligned = data.rio.reproject_match(master_obj, resampling=method)
            # 保存
            aligned.astype("float32").rio.to_raster(out_path, compress="lzw")
        except Exception as e:
            tqdm.write(f"❌ 错误: {in_path} -> {e}")

def run_resampling():
    print("📖 正在加载大师模板空间属性...")
    master = rioxarray.open_rasterio(MASTER_TIF)

    # ==========================================
    # 阶段 A: 静态特征处理
    # ==========================================
    print("\n🏔️ 阶段 A: 开始处理静态特征...")
    static_out = os.path.join(OUTPUT_ROOT, "Static")
    static_tasks = []

    for name, p in PATHS["STATIC"].items():
        if os.path.exists(p):
            out_file = os.path.join(static_out, f"{name}_5km.tif")
            t_type = 'copy' if name == 'Roads' else ('nearest' if name == 'CLCD' else 'bilinear')
            static_tasks.append((p, out_file, t_type))

    if static_tasks:
        for in_p, out_p, t_type in tqdm(static_tasks, desc="静态特征处理", unit="图"):
            process_single_file(in_p, out_p, master, t_type)
        print("✅ 静态特征重采样与对齐全部成功！")

    # ==========================================
    # 阶段 B: NDVI 每日动态数据处理
    # ==========================================
    print("\n🌿 阶段 B: 开始处理每日 NDVI...")
    ndvi_in_dir = PATHS["DAILY"]["NDVI"]
    ndvi_out = os.path.join(OUTPUT_ROOT, "NDVI_Daily")

    if os.path.exists(ndvi_in_dir):
        ndvi_files = [f for f in os.listdir(ndvi_in_dir) if f.endswith(".tif")]
        if ndvi_files:
            for f in tqdm(ndvi_files, desc="NDVI 逐日重采样", unit="图"):
                in_p = os.path.join(ndvi_in_dir, f)
                out_p = os.path.join(ndvi_out, f)
                process_single_file(in_p, out_p, master, 'bilinear')
            print("✅ 每日 NDVI 强制重采样成功！")
    else:
        print(f"⚠️ 找不到 NDVI 目录: {ndvi_in_dir}")

    # ==========================================
    # 阶段 C: ERA5 每小时气象特征处理 (按变量分子任务)
    # ==========================================
    print("\n🛰️ 阶段 C: 开始处理每小时气象变量 (ERA5)...")
    era5_root = PATHS["HOURLY"]["ERA5"]
    era5_vars = ['blh', 'd2m', 'lcc', 'sp', 't2m', 'tcc', 'u10', 'v10', 'rh']

    for var_folder in era5_vars:
        var_path = os.path.join(era5_root, var_folder)
        if not os.path.exists(var_path):
            continue

        # 1. 快速扫描当前变量下的所有文件
        var_tasks = []
        for root, dirs, files in os.walk(var_path):
            for f in files:
                if f.endswith(".tif"):
                    rel_path = os.path.relpath(root, var_path)
                    in_p = os.path.join(root, f)
                    out_p = os.path.join(OUTPUT_ROOT, "ERA5", var_folder, rel_path, f)
                    var_tasks.append((in_p, out_p))

        # 2. 为当前变量启动专属进度条
        if var_tasks:
            # 这里的 desc 会显示为类似 "ERA5: t2m"
            for in_p, out_p in tqdm(var_tasks, desc=f"ERA5: {var_folder:<4}", unit="图"):
                process_single_file(in_p, out_p, master, 'bilinear')
            print(f"✅ 变量 [{var_folder}] 覆盖重采样完毕！")

    print("\n🎉🎉 全部特征数据的重采样与强制覆盖工程圆满完成！")


if __name__ == "__main__":
    run_resampling()