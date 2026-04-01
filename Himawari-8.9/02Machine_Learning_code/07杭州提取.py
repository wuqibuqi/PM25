import os
import glob
import numpy as np
import rasterio
from rasterio.mask import mask
import fiona
from tqdm import tqdm
import gc

# ================= 1. 路径与核心配置 =================
# 输入路径：你反演结果的根目录
INPUT_ROOT = r"D:\1document\Graduation Thesis\01Code\DATA\PM25_Retrieval_Results_Seamless"

# SHP路径：你刚刚提供的杭州市边界
HZ_SHP_PATH = r"D:\1document\Graduation Thesis\Data\审图号：GS（2024）0650号\杭州市_市\杭州市_市.shp"

# 输出路径：建议单独存放提取后的结果
HZ_OUTPUT_DIR = r"D:\1document\Graduation Thesis\01Code\DATA\Hangzhou_Monthly_Analysis"

# 设定处理年份
YEARS = ["2020", "2021", "2022", "2023"]

os.makedirs(HZ_OUTPUT_DIR, exist_ok=True)


# ================= 2. 执行提取与聚合 =================

def extract_and_average_hz():
    # 1. 加载杭州边界矢量
    print(f"🗺️ 正在载入杭州市边界文件...")
    try:
        with fiona.open(HZ_SHP_PATH, "r") as shapefile:
            # 提取几何图形用于裁剪
            shapes = [feature["geometry"] for feature in shapefile]
            # 获取原始坐标系信息
            shp_crs = shapefile.crs
    except Exception as e:
        print(f"❌ 无法读取SHP文件，请检查路径或编码: {e}")
        return

    for year in YEARS:
        year_in_path = os.path.join(INPUT_ROOT, year)
        year_out_path = os.path.join(HZ_OUTPUT_DIR, year)

        if not os.path.exists(year_in_path):
            print(f"⏩ 跳过 {year} 年 (未找到反演目录)")
            continue

        os.makedirs(year_out_path, exist_ok=True)

        # 逐月循环 01-12
        for month in range(1, 13):
            mm_str = f"{month:02d}"
            month_path = os.path.join(year_in_path, mm_str)

            if not os.path.exists(month_path):
                continue

            # 搜索该月份下所有天（Day）的所有小时（Hour）TIF文件
            # 结构示例：.../2020/01/01/PM25_Seamless_20200101_0000.tif
            search_pattern = os.path.join(month_path, "**", "*.tif")
            all_tifs = glob.glob(search_pattern, recursive=True)

            if not all_tifs:
                continue

            print(f"📅 正在计算 {year}年{mm_str}月... (样本数: {len(all_tifs)})")

            # 初始化累加器
            sum_array = None
            count_array = None
            out_meta = None

            # 遍历该月所有小时数据进行累加
            for tif in tqdm(all_tifs, desc=f"Progress {mm_str}", leave=False):
                try:
                    with rasterio.open(tif) as src:
                        # 核心步骤：按边界裁剪 (crop=True 自动收缩至杭州矩形范围)
                        # nodata 设为 np.nan 方便后续计算平均值
                        hz_data, hz_transform = mask(src, shapes, crop=True, nodata=np.nan)

                        # 初始化矩阵尺寸 (第一个有效文件时执行)
                        if sum_array is None:
                            sum_array = np.zeros_like(hz_data[0], dtype=np.float32)
                            count_array = np.zeros_like(hz_data[0], dtype=np.int32)

                            # 更新元数据用于保存输出
                            out_meta = src.meta.copy()
                            out_meta.update({
                                "driver": "GTiff",
                                "height": hz_data.shape[1],
                                "width": hz_data.shape[2],
                                "transform": hz_transform,
                                "nodata": np.nan,
                                "dtype": 'float32'
                            })

                        # 统计有效像元 (非NaN值)
                        valid_mask = ~np.isnan(hz_data[0])
                        sum_array[valid_mask] += hz_data[0][valid_mask]
                        count_array[valid_mask] += 1
                except Exception:
                    continue

            # 计算月均值并保存
            if sum_array is not None:
                # 矩阵运算：总和 / 计数 = 平均
                with np.errstate(divide='ignore', invalid='ignore'):
                    monthly_mean = np.where(count_array > 0, sum_array / count_array, np.nan)

                out_name = f"HZ_PM25_MonthlyMean_{year}{mm_str}.tif"
                out_save_path = os.path.join(year_out_path, out_name)

                with rasterio.open(out_save_path, "w", **out_meta) as dest:
                    dest.write(monthly_mean.astype(np.float32), 1)

                # 清理内存
                del sum_array, count_array, monthly_mean
                gc.collect()

    print(f"\n✨ 杭州市逐月平均数据提取完成！结果存放在: {HZ_OUTPUT_DIR}")


if __name__ == "__main__":
    extract_and_average_hz()