import os
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

# ================= 1. 路径与配置 (请核对路径) =================
# 地面实测真值表 (索引表)
GROUND_CSV = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\Output\Ground_Value_2020_2023_More.csv"

# Light 反演结果根目录 (D盘)
Light_ROOT = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM25_Retrieval_Results_Seamless")
# RF 反演结果根目录 (E盘)
RF_ROOT = Path(r"E:\PM25_Retrieval_Results")

# 输出的总验证表 (双模型整合)
FINAL_VAL_CSV = r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\对比反演结果.csv"


# ================= 2. 核心矩阵提取函数 =================
def fast_extract_val(tif_path, lons, lats):
    """利用 NumPy 矩阵索引极速提取"""
    if not tif_path.exists():
        return np.full(len(lons), np.nan)

    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            # 批量经纬度转行列号
            rows, cols = rasterio.transform.rowcol(src.transform, lons, lats)
            rows, cols = np.array(rows), np.array(cols)

            # 边界检查
            valid_mask = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)

            res = np.full(len(lons), np.nan)
            if valid_mask.any():
                vals = data[rows[valid_mask], cols[valid_mask]]
                # 清洗异常值 (0-1000 ug/m3 属于正常范围)
                vals = np.where((vals >= 0) & (vals < 1000), vals, np.nan)
                res[valid_mask] = vals
        return res
    except Exception:
        return np.full(len(lons), np.nan)


# ================= 3. 主提取流程 =================
def run_combined_extraction():
    print("📖 正在加载地面实测数据...")
    df = pd.read_csv(GROUND_CSV)
    df['UTC_Time'] = pd.to_datetime(df['UTC_Time'])

    # 结果容器
    combined_results = []

    # 按时间分组 (核心提速：每小时只寻址一次文件夹)
    time_groups = df.groupby('UTC_Time')

    print(f"🚀 开始同步提取 Light 和 RF 结果 (共 {len(time_groups)} 个时点)...")

    for utc_time, group in tqdm(time_groups):
        # --- A. 构建 Light 路径 ---
        # 结构: ROOT/年/月/日/PM25_Seamless_YYYYMMDD_HH00.tif
        Light_dir = Light_ROOT / utc_time.strftime('%Y') / utc_time.strftime('%m') / utc_time.strftime('%d')
        Light_file = Light_dir / f"PM25_Seamless_{utc_time.strftime('%Y%m%d_%H')}00.tif"

        # --- B. 构建 RF 路径 ---
        # 结构: ROOT/年/月/日/PM25_Retrieved_YYYYMMDD_HH00_UTC.tif
        rf_dir = RF_ROOT / utc_time.strftime('%Y') / utc_time.strftime('%m') / utc_time.strftime('%d')
        rf_file = rf_dir / f"PM25_Retrieved_{utc_time.strftime('%Y%m%d_%H')}00_UTC.tif"

        # --- C. 执行提取 ---
        lons = group['Longitude'].values
        lats = group['Latitude'].values

        current_group = group.copy()
        current_group['Light_Value'] = fast_extract_val(Light_file, lons, lats)
        current_group['RF_Value'] = fast_extract_val(rf_file, lons, lats)

        combined_results.append(current_group)

    # 合并所有数据
    final_df = pd.concat(combined_results, ignore_index=True)

    # 【重要】只保留两个模型都有提取结果的行，用于公平对比
    # 如果你想保留哪怕只有一个模型有结果的行，可以将 'how' 改为 'any'
    final_df = final_df.dropna(subset=['Light_Value', 'RF_Value'], how='all')

    # 保存总表
    final_df.to_csv(FINAL_VAL_CSV, index=False, encoding='utf-8-sig')

    print(f"\n🎉 提取整合完毕！")
    print(f"📂 输出路径: {FINAL_VAL_CSV}")
    print(f"📊 总有效对比样本数: {len(final_df)} 条")

    # 打印一个简单的初步统计
    print("\n📝 模型表现预览 (均值):")
    print(f"   - 地面实测均值: {final_df['PM25_5030'].mean():.2f}")
    print(f"   - Light 反演均值: {final_df['Light_Value'].mean():.2f}")
    print(f"   - RF 反演均值: {final_df['RF_Value'].mean():.2f}")


if __name__ == "__main__":
    run_combined_extraction()