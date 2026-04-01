#从ERA5下载的是每隔半年的数据zip，要把它转换成年月日结构储存的nc文件
import xarray as xr
import os
import zipfile
import shutil
import glob
from tqdm import tqdm
import pandas as pd

# --- 配置路径 ---
BASE_DIR = r"E:\ERA5_Data\source data"
TEMP_EXTRACT_DIR = os.path.join(BASE_DIR, "temp_raw")


def process_era5_to_hourly():
    # 1. 寻找并解压所有 ZIP 文件
    zip_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.zip')]
    if not zip_files:
        print("❓ 未在目录下找到 ZIP 文件，请确认路径。")
        return

    if os.path.exists(TEMP_EXTRACT_DIR):
        shutil.rmtree(TEMP_EXTRACT_DIR)
    os.makedirs(TEMP_EXTRACT_DIR)

    for zf in zip_files:
        print(f"📦 正在解压: {zf}...")
        # 修正逻辑：为每个zip创建子目录，防止内部 nc 文件重名覆盖
        extract_sub_dir = os.path.join(TEMP_EXTRACT_DIR, zf.replace(".zip", ""))
        with zipfile.ZipFile(os.path.join(BASE_DIR, zf), 'r') as zip_ref:
            zip_ref.extractall(extract_sub_dir)

    # 使用 glob 递归寻找所有子目录下的 nc 文件
    nc_files = glob.glob(os.path.join(TEMP_EXTRACT_DIR, "**", "*.nc"), recursive=True)
    print(f"🚀 开始按小时拆分数据 (共 {len(nc_files)} 个原始文件)...")

    for nc_file in nc_files:
        with xr.open_dataset(nc_file) as ds:
            # --- 核心修正：自动寻找时间维度 ---
            time_var = None
            for var in ['time', 'valid_time', 'TIME', 't']:
                if var in ds.coords or var in ds.dims:
                    time_var = var
                    break

            if time_var is None:
                print(f"❌ 错误：在文件 {os.path.basename(nc_file)} 中找不到时间维度！")
                print(f"包含的维度有: {list(ds.dims)}")
                continue

            # 使用找到的时间变量提取数据
            time_values = ds[time_var].values

            for t in tqdm(time_values, desc=f"拆分 {os.path.basename(nc_file)}"):
                ts = pd.to_datetime(t)
                year_str = ts.strftime('%Y')
                month_str = ts.strftime('%m')
                day_hour_str = ts.strftime('%Y%m%d_%H%M')

                target_dir = os.path.join(BASE_DIR, year_str, month_str)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                output_path = os.path.join(target_dir, f"ERA5_{day_hour_str}.nc")

                if os.path.exists(output_path):
                    continue

                # 使用动态的时间变量名进行筛选
                hourly_ds = ds.sel({time_var: t})
                hourly_ds.to_netcdf(output_path, format="NETCDF4", engine="netcdf4")

    print("🧹 正在清理临时文件...")
    shutil.rmtree(TEMP_EXTRACT_DIR)
    print("✅ 全部处理完成！")


if __name__ == "__main__":
    process_era5_to_hourly()