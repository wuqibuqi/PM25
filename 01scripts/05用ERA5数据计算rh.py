import os
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

# --- 1. 路径配置 ---
base_path = Path(r'E:\ERA5_TIF')
t2m_root = base_path / 't2m'
d2m_root = base_path / 'd2m'
rh_root = base_path / 'rh'


def calculate_rh_celsius(t2m_c, d2m_c):
    """
    使用 August-Roche-Magnus 公式计算相对湿度
    针对摄氏度数据优化，并严格控制异常值
    """
    mask = np.isnan(t2m_c) | np.isnan(d2m_c) | (t2m_c < -80) | (t2m_c > 60)
    A = 17.625
    B = 243.04

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        numerator = np.exp((A * d2m_c) / (B + d2m_c))
        denominator = np.exp((A * t2m_c) / (B + t2m_c))
        rh = 100 * (numerator / denominator)

    rh[mask] = np.nan
    rh = np.clip(rh, 0, 100)
    return rh.astype(np.float32)


def main():
    if not t2m_root.exists():
        print(f"错误: 找不到 t2m 文件夹路径: {t2m_root}")
        return

    # 递归搜索所有 t2m.tif 文件
    tasks = list(t2m_root.rglob('*_t2m.tif'))
    print(f"检测到共 {len(tasks)} 个待处理文件。开始断点续传检测...")

    # 记录跳过的文件数量
    skip_count = 0

    for t2m_path in tqdm(tasks, desc="RH 计算进度", unit="file"):
        # 1. 定位 d2m 文件和构造 rh 输出路径
        d2m_path = Path(str(t2m_path).replace('t2m', 'd2m'))
        out_path = Path(str(t2m_path).replace('t2m', 'rh'))

        # --- 【核心：断点续传逻辑】 ---
        # 如果对应的 rh.tif 已经存在，则直接跳过
        if out_path.exists():
            skip_count += 1
            continue
        # -----------------------------

        if not d2m_path.exists():
            tqdm.write(f"跳过: 找不到对应的 d2m 文件 {d2m_path.name}")
            continue

        try:
            # 确保输出目录存在
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(t2m_path) as src_t:
                t_arr = src_t.read(1)
                profile = src_t.profile

                with rasterio.open(d2m_path) as src_d:
                    td_arr = src_d.read(1)

                    # 执行 RH 计算
                    rh_arr = calculate_rh_celsius(t_arr, td_arr)

                    # 更新元数据
                    profile.update(dtype=rasterio.float32, nodata=np.nan)

                    # 写入结果
                    with rasterio.open(out_path, 'w', **profile) as dst:
                        dst.write(rh_arr, 1)

        except Exception as e:
            tqdm.write(f"处理文件 {t2m_path.name} 时出错: {e}")

    print(f"✅ 处理完成！本次共跳过 {skip_count} 个已存在的文件。")


if __name__ == "__main__":
    main()