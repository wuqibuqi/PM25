#给ERA5数据按照十个变量分类并转化为tif
import xarray as xr
import os
import glob
from tqdm import tqdm
import rioxarray

# --- 1. 配置路径 ---
INPUT_ROOT = r"E:\ERA5_Data\source data"#ERA5数据储存地址
OUTPUT_ROOT = r"E:\ERA5_TIF"#转换后地址

# --- 2. 10个变量及其换算逻辑 ---
# 键为nc内部变量名，值为你希望的文件夹简称
VARIABLES_CONFIG = {
    't2m': 't2m',  # 2m温度
    'd2m': 'd2m',  # 2m露点
    'u10': 'u10',  # 10m风速U分量
    'v10': 'v10',  # 10m风速V分量
    'sp': 'sp',  # 地面气压
    'blh': 'blh',  # 边界层高度
    'tp': 'tp',  # 总降水
    'tcc': 'tcc',  # 总云量
    'lcc': 'lcc',  # 低云量
    'ssrd': 'ssrd'  # 太阳辐射
}


def convert_separated_vars_to_tif():
    nc_files = glob.glob(os.path.join(INPUT_ROOT, "**", "*.nc"), recursive=True)
    if not nc_files:
        print("❓ 未找到 .nc 文件。")
        return

    print(f"🚀 开始分离转换（支持断点续传模式）...")

    for nc_path in tqdm(nc_files, desc="总进度"):
        try:
            relative_date_path = os.path.relpath(os.path.dirname(nc_path), INPUT_ROOT)
            base_name = os.path.basename(nc_path).replace(".nc", "")

            # --- 【新增：断点续传检查逻辑】 ---
            # 检查该 NC 文件对应的所有变量 TIF 是否都已经生成
            all_done = True
            for var_key, folder_name in VARIABLES_CONFIG.items():
                output_path = os.path.join(OUTPUT_ROOT, folder_name, relative_date_path, f"{base_name}_{var_key}.tif")
                if not os.path.exists(output_path):
                    all_done = False
                    break

            if all_done:
                # 如果全部都存在，直接跳过这个文件，不进行任何操作
                continue
                # ----------------------------------

            # 只有在需要更新或补全时才打开数据集
            ds = xr.open_dataset(nc_path, engine="netcdf4", mask_and_scale=True)
            ds = ds.rio.write_crs("EPSG:4326")

            for var_key, folder_name in VARIABLES_CONFIG.items():
                if var_key in ds.data_vars:
                    var_target_dir = os.path.join(OUTPUT_ROOT, folder_name, relative_date_path)
                    output_path = os.path.join(var_target_dir, f"{base_name}_{var_key}.tif")

                    # --- 【单变量断点检查】 ---
                    if os.path.exists(output_path):
                        continue  # 该变量已存在，跳过此变量

                    if not os.path.exists(var_target_dir):
                        os.makedirs(var_target_dir)

                    data_array = ds[var_key]

                    # 降维处理
                    time_dim = next((d for d in ['time', 'valid_time', 'TIME'] if d in data_array.dims), None)
                    if time_dim:
                        data_array = data_array.isel({time_dim: 0})

                    # 物理单位换算
                    if var_key in ['t2m', 'd2m']:
                        data_array = data_array - 273.15
                    elif var_key == 'tp':
                        data_array = data_array * 1000

                    # 导出
                    data_array.rio.to_raster(output_path, compress='LZW')

            ds.close()
        except Exception as e:
            print(f"❌ 转换 {nc_path} 失败: {e}")


if __name__ == "__main__":
    convert_separated_vars_to_tif()
    print(f"✅ 转换完成！所有变量已按简称分文件夹存储于: {OUTPUT_ROOT}")