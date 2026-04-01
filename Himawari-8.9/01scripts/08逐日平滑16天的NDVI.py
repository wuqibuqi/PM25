import os
import re
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from datetime import datetime, timedelta
import pandas as pd

# --- 1. 配置路径 ---
INPUT_DIR = r"E:\NDVI_16Day"
OUTPUT_DIR = r"E:\Static_Features\NDVI_Daily"
# 你的研究区范围 [北, 南, 东, 西]
NORTH, SOUTH, EAST, WEST = 36, 27, 123, 114

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def get_date_from_filename(fname):
    # 直接寻找 _NDVI_ 后面跟着的 8 位连续数字
    # 不需要转义 20，直接写即可
    match = re.search(r'_NDVI_(\d{8})T', fname)
    if match:
        # match.group(1) 提取的就是那 8 位数字字符串
        return datetime.strptime(match.group(1), '%Y%m%d')
    return None


def process_ndvi_daily():
    # 1. 获取并排序所有文件
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.tif')]
    file_info = []
    for f in files:
        dt = get_date_from_filename(f)
        if dt:
            file_info.append({'date': dt, 'path': os.path.join(INPUT_DIR, f)})

    df = pd.DataFrame(file_info).sort_values('date').reset_index(drop=True)
    print(f"📅 检测到 {len(df)} 期 NDVI 数据，准备开始插值...")

    # 裁剪用的几何体
    roi_geom = [box(WEST, SOUTH, EAST, NORTH)]

    # 2. 遍历相邻的两期数据进行线性插值
    for i in range(len(df) - 1):
        t1, t2 = df.iloc[i]['date'], df.iloc[i + 1]['date']
        path1, path2 = df.iloc[i]['path'], df.iloc[i + 1]['path']

        days_diff = (t2 - t1).days
        print(f"正在处理区间: {t1.date()} -> {t2.date()} ({days_diff}天)")

        with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
            # 裁剪并读取数据
            out_img1, out_trans = mask(src1, roi_geom, crop=True)
            out_img2, _ = mask(src2, roi_geom, crop=True)

            # MODIS NDVI 通常有 0.0001 的比例因子
            data1 = out_img1[0].astype('float32') * 0.0001
            data2 = out_img2[0].astype('float32') * 0.0001

            # 生成中间每一天的图像
            for d in range(days_diff):
                current_date = t1 + timedelta(days=d)
                # 线性插值公式: y = y1 + (y2 - y1) * (t - t1) / (t2 - t1)
                weight = d / days_diff
                interp_data = data1 + (data2 - data1) * weight

                # 保存结果
                out_name = f"NDVI_{current_date.strftime('%Y%m%d')}.tif"
                out_path = os.path.join(OUTPUT_DIR, out_name)

                meta = src1.meta.copy()
                meta.update({
                    "driver": "GTiff",
                    "height": interp_data.shape[0],
                    "width": interp_data.shape[1],
                    "transform": out_trans,
                    "dtype": 'float32',
                    "count": 1,
                    "compress": 'lzw',
                    "nodata": -9999
                })

                with rasterio.open(out_path, "w", **meta) as dest:
                    dest.write(interp_data.astype('float32'), 1)
    # --- 【核心修正：单独处理全系列最后一天 (t_final)】 ---
    last_row = df.iloc[-1]
    last_date = last_row['date']
    last_path = last_row['path']
    print(f"🏁 正在处理全系列最后一天: {last_date.date()}")

    with rasterio.open(last_path) as src_last:
        out_img_last, out_trans_last = mask(src_last, roi_geom, crop=True)
        # 别忘了乘以比例因子
        final_data = out_img_last[0].astype('float32') * 0.0001

        out_name = f"NDVI_{last_date.strftime('%Y%m%d')}.tif"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        meta = src_last.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": final_data.shape[0],
            "width": final_data.shape[1],
            "transform": out_trans_last,
            "dtype": 'float32',
            "count": 1,
            "compress": 'lzw',
            "nodata": -9999
        })

        with rasterio.open(out_path, "w", **meta) as dest:
            dest.write(final_data.astype('float32'), 1)
if __name__ == "__main__":
    process_ndvi_daily()