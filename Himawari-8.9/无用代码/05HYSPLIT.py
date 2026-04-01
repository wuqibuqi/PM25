import subprocess
import os
import pandas as pd
from pathlib import Path

# ==============================================================
# --- 1. 路径与参数配置 (请务必核对) ---
# ==============================================================
HYSPLIT_DIR = Path(r"D:\APPData\HYSPLIT")  # HYSPLIT 安装目录
ARL_DIR = Path(r"D:\1document\Graduation Thesis\01Code\DATA\ARL_Data")
CSV_PATH = Path(
    r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\PM2.5_Meteorology_Study_3day\Refined_Golden_Cases_Selection.csv")
OUT_DIR = Path(r"D:\1document\Graduation Thesis\01Code\DATA\HYSPLIT_Results")

EXEC_PATH = HYSPLIT_DIR / "exec" / "hyts_std.exe"
WORKING_DIR = HYSPLIT_DIR / "working"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 杭州受体点坐标
LAT, LON = 30.27, 120.15
# 模拟高度 (通常论文做 100m, 500m, 1000m 三个层次)
HEIGHTS = [100, 500, 1000]
BACK_HOURS = -72  # 后向追踪 72 小时

# ==============================================================
# --- 2. 案例与 ARL 文件映射表 ---
# ==============================================================
# 这里的逻辑是根据你下载的文件名，手动映射到 CSV 的 4 个案例索引上
ARL_MAP = {
    0: ["gdas1.dec20.w3", "gdas1.dec20.w4"],
    1: ["gdas1.jan22.w5", "gdas1.feb22.w1"],
    2: ["gdas1.jan23.w3", "gdas1.jan23.w4"],
    3: ["gdas1.dec23.w4", "gdas1.dec23.w5"]
}


# ==============================================================
# --- 3. 核心计算引擎 ---
# ==============================================================

def run_hysplit_batch():
    df = pd.read_csv(CSV_PATH)

    for idx, row in df.iterrows():
        case_name = f"Case_{idx}_{row['冬季案例']}"
        # 转换时间格式为 HYSPLIT 要求的 YY MM DD HH
        peak_t = pd.to_datetime(row['峰值时刻'])
        start_time_str = peak_t.strftime('%y %m %d %H')

        for h in HEIGHTS:
            output_name = f"tdump_{idx}_H{h}"
            needed_arl = ARL_MAP.get(idx, [])

            # --- 构建 CONTROL 文件内容 ---
            lines = [
                start_time_str,  # 1. 起始时间
                "1",  # 2. 释放点个数
                f"{LAT} {LON} {h}",  # 3. 纬度 经度 高度
                str(BACK_HOURS),  # 4. 追踪时长
                "0",  # 5. 垂直运动方法
                "10000.0",  # 6. 顶层限制
                str(len(needed_arl))  # 7. ARL 文件数量
            ]
            for arl in needed_arl:
                lines.append(str(ARL_DIR) + os.sep)  # 8. ARL 路径
                lines.append(arl)  # 9. ARL 文件名

            lines.append(str(OUT_DIR) + os.sep)  # 10. 输出目录
            lines.append(output_name)  # 11. 输出文件名

            # 写入 WORKING 目录下的 CONTROL 文件
            with open(WORKING_DIR / "CONTROL", "w") as f:
                f.write("\n".join(lines) + "\n")

            # --- 调用 EXE 计算 ---
            print(f"🚀 正在计算: {case_name} | 高度: {h}m...")
            result = subprocess.run(
                str(EXEC_PATH),
                cwd=str(WORKING_DIR),
                shell=True,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print(f"✅ 完成: {output_name}")
            else:
                print(f"❌ 失败: {case_name}\n{result.stderr}")


if __name__ == "__main__":
    run_hysplit_batch()
    print(f"\n✨ 所有轨迹计算完毕！结果已保存至: {OUT_DIR}")