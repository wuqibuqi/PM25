import os
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import re

# --- 1. 路径与配置 ---
INPUT_ROOT = Path(r"E:\PM25_Retrieval_Results")
OUTPUT_ROOT = Path(r"E:\PM25_Averages")

SEASON_MAP = {
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11],
    'Winter': [12, 1, 2]
}


def get_date_info(filename):
    # 修正后的正则表达式：去掉多余的反斜杠
    match = re.search(r'(202[0-3])(\d{2})(\d{2})', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def save_tif(data, template_profile, out_path):
    template_profile.update(dtype=rasterio.float32, nodata=np.nan, count=1, compress='lzw')
    with rasterio.open(out_path, 'w', **template_profile) as dst:
        dst.write(data.astype(np.float32), 1)


def main():
    if not INPUT_ROOT.exists():
        print(f"❌ 错误：路径不存在 {INPUT_ROOT}")
        return

    all_files = list(INPUT_ROOT.rglob("*.tif"))
    if not all_files:
        print("❓ 未找到任何 .tif 文件")
        return

    print(f"🔍 检索到共 {len(all_files)} 个原始文件")

    # --- 调试步骤：检查前 5 个文件是否能解析日期 ---
    print("🧪 正在测试文件名解析逻辑...")
    for i in range(min(5, len(all_files))):
        y, m = get_date_info(all_files[i].name)
        print(f"文件名: {all_files[i].name}  => 解析结果: 年={y}, 月={m}")

    stats = {}
    with rasterio.open(all_files[0]) as src:
        template_profile = src.profile
        shape = src.shape

    # --- 2. 遍历累加阶段 ---
    for f_path in tqdm(all_files, desc="数据累加进度"):
        year, month = get_date_info(f_path.name)
        if year is None:
            continue  # 如果解析失败，跳过

        season = next((s for s, months in SEASON_MAP.items() if month in months), None)
        keys = [('Month', year, month), ('Season', year, season)]

        try:
            with rasterio.open(f_path) as src:
                data = src.read(1)
                data[(data < 0) | (data > 1000)] = np.nan

                for key in keys:
                    if key[2] is None: continue
                    if key not in stats:
                        stats[key] = [np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.int32)]

                    valid_mask = ~np.isnan(data)
                    stats[key][0][valid_mask] += data[valid_mask]
                    stats[key][1][valid_mask] += 1
        except Exception as e:
            pass

    if not stats:
        print("❌ 统计字典为空！可能是文件名中的年份/月份没有被正确解析。")
        return

    # --- 3. 计算均值并导出阶段 ---
    print(f"\n💾 正在计算均值并导出 {len(stats)} 个 TIF 文件...")
    for key, (sum_mat, count_mat) in tqdm(stats.items(), desc="导出进度"):
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_mat = np.where(count_mat > 0, sum_mat / count_mat, np.nan)

        type_str, yr, period = key
        out_dir = OUTPUT_ROOT / type_str / str(yr)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"YRD_PM25_{type_str}_{yr}_{period}.tif"
        save_tif(mean_mat, template_profile, out_dir / out_name)

    print(f"✅ 完成！结果保存至: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()