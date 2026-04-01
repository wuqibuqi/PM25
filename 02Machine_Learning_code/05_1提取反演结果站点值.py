import os
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

# ================= 1. 路径配置 =================
# 你的地面实测真值表 (刚刚扩充完的那个)
GROUND_CSV = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\Output\Ground_Value_2020_2023_More.csv"
# 反演结果 TIF 根目录
INVERSION_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM25_Retrieval_Results_Seamless")
# 输出的验证对比表路径
OUTPUT_VAL_CSV = r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\反演结果站点提取表.csv"


# ================= 2. 矩阵提取核心函数 =================
def fast_extract(tif_path, lons, lats):
    """利用 NumPy 矩阵索引极速提取"""
    if not tif_path.exists():
        return np.full(len(lons), np.nan)

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        # 批量经纬度转行列号
        rows, cols = rasterio.transform.rowcol(src.transform, lons, lats)
        rows, cols = np.array(rows), np.array(cols)

        # 边界安全检查
        valid_mask = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)

        res = np.full(len(lons), np.nan)
        if valid_mask.any():
            # 提取值并清洗无效大值
            vals = data[rows[valid_mask], cols[valid_mask]]
            vals = np.where(vals < 1000, vals, np.nan)  # 假设浓度不会超过1000
            res[valid_mask] = vals
    return res


# ================= 3. 执行提取流程 =================
def start_extraction():
    print("📖 正在加载地面实测数据...")
    df = pd.read_csv(GROUND_CSV)

    # 确保时间列是日期格式 (使用 UTC 时间对接文件名)
    df['UTC_Time'] = pd.to_datetime(df['UTC_Time'])

    # 结果存储
    extracted_results = []

    # 按时间分组处理 (每一小时只需打开一次 TIF 文件，大幅提速)
    time_groups = df.groupby('UTC_Time')

    print(f"🚀 开始从嵌套文件夹提取反演结果 (共 {len(time_groups)} 个时点)...")

    for utc_time, group in tqdm(time_groups):
        # 构建年月日嵌套路径
        # 结构示例: INVERSION_ROOT / 2020 / 01 / 01 / PM25_Seamless_20200101_0000.tif
        sub_dir = INVERSION_ROOT / utc_time.strftime('%Y') / utc_time.strftime('%m') / utc_time.strftime('%d')
        file_name = f"PM25_Seamless_{utc_time.strftime('%Y%m%d_%H')}00.tif"
        tif_path = sub_dir / file_name

        # 提取当前小时所有站点的反演值
        lons = group['Longitude'].values
        lats = group['Latitude'].values

        # 将反演值存入该组数据
        group_with_val = group.copy()
        group_with_val['Inversion_Value'] = fast_extract(tif_path, lons, lats)

        extracted_results.append(group_with_val)

    # 合并所有结果
    final_df = pd.concat(extracted_results, ignore_index=True)

    # 剔除那些没对上 TIF 文件的空行 (NaN)
    final_df = final_df.dropna(subset=['Inversion_Value'])

    # 保存新表
    final_df.to_csv(OUTPUT_VAL_CSV, index=False, encoding='utf-8-sig')
    print(f"\n🎉 提取完成！验证对比表已保存至: {OUTPUT_VAL_CSV}")
    print(f"📈 有效配对样本数: {len(final_df)} 条")


if __name__ == "__main__":
    start_extraction()