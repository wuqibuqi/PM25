# 将从jaxa下载的数据从nc转换为tif
import os
import xarray as xr
import rioxarray
from tqdm import tqdm  # 导入进度条神器

# --- 1. 路径设置 ---
source_dir = r"E:\data"#葵花nc源数据存放路径
target_dir = r"E:\Himawari-8_TIFF_24h"#葵花tif数据存放路径

# --- 2. 筛选与裁剪条件 ---
# 筛选北京时间 08:00 - 17:00 (即 UTC 0000 - 0900)
valid_utc_times = [f"{i:02d}00" for i in range(24)]

# 长三角及华东区域裁剪坐标 (建议保持开启以节省空间)
lon_min, lon_max = 114.0, 123.0
lat_min, lat_max = 27.0, 36.0


def process_nc_to_tif():
    print("正在扫描需要转换的文件，请稍候...")
    tasks = []

    # --- 第一阶段：收集所有需要处理的任务 ---
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".nc")and '1HARP' in file:
                parts = file.split('_')
                if len(parts) >= 4:
                    time_str = parts[2]
                    if time_str in valid_utc_times:
                        source_path = os.path.join(root, file)

                        # 构建目标路径
                        rel_path = os.path.relpath(root, source_dir)
                        current_target_dir = os.path.join(target_dir, rel_path)
                        os.makedirs(current_target_dir, exist_ok=True)

                        tif_name = file.replace('.nc', '.tif')
                        target_file = os.path.join(current_target_dir, tif_name)

                        # 如果目标文件不存在，则加入待处理任务列表
                        if not os.path.exists(target_file):
                            tasks.append((source_path, target_file, file))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("🎉 没有发现需要处理的新文件，所有数据均已转换完毕！")
        return

    print(f"扫描完毕！共发现 {total_tasks} 个待转换文件。开始全速处理：")

    # --- 第二阶段：执行转换并显示进度条 ---
    # tqdm(tasks) 会自动在控制台生成一个动态更新的进度条
    for source_path, target_file, file_name in tqdm(tasks, desc="TIFF 转换进度", unit="文件"):
        try:
            # 1. 读取数据 (加入 decode_timedelta=False 消除版本警告)
            ds = xr.open_dataset(source_path, decode_timedelta=False)

            # 2. 提取 AOD 变量
            da = ds['AOT_Merged']

            # 【过滤代码添加在这里】：只保留大于等于 0 的有效值，负数将被替换为 NaN (空值)
            da = da.where(da >= 0)

            # 3. 空间参考与裁剪
            da = da.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
            da.rio.write_crs("epsg:4326", inplace=True)
            da = da.rio.clip_box(minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max)

            # 4. 导出保存
            da.rio.to_raster(target_file)
            ds.close()

        except Exception as e:
            # 使用 tqdm.write 可以在不打断进度条显示的情况下打印报错信息
            tqdm.write(f"❌ 处理 {file_name} 时报错: {e}")


if __name__ == "__main__":
    process_nc_to_tif()