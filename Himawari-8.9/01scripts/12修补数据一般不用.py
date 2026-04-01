import subprocess
import os
import sys
import time
from tqdm import tqdm

# --- 1. 配置脚本绝对路径 ---
# 请核对你的实际路径，确保无误
BASE_SCRIPTS_DIR = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\scripts"
ML_CODE_DIR = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Machine_Learning_code"

WORKFLOW = [
    {"name": "提取 NC 文件", "path": os.path.join(BASE_SCRIPTS_DIR, "04解压era5的zip数据按时间储存.py")},
    {"name": "NC 转 TIF (分类)", "path": os.path.join(BASE_SCRIPTS_DIR, "05era5数据分类转化为tif.py")},
    {"name": "计算相对湿度 (RH)", "path": os.path.join(BASE_SCRIPTS_DIR, "05用ERA5数据计算rh.py")},
    {"name": "数据重采样", "path": os.path.join(BASE_SCRIPTS_DIR, "10重采样除土地利用外的数据.py")},
    {"name": "PM2.5 反演制图", "path": os.path.join(ML_CODE_DIR, "03两阶段反演模型.py")}
]


def run_step(script_name, script_path):
    print(f"\n▶️  正在执行: {script_name}")
    print(f"📂 路径: {script_path}")

    start_time = time.time()
    try:
        # 使用当前环境的 python 解释器执行
        # check=True 表示如果脚本报错，主控脚本会立即停止，方便你排查
        result = subprocess.run([sys.executable, script_path], check=True)
        end_time = time.time()
        print(f"✅ {script_name} 执行成功！耗时: {(end_time - start_time) / 60:.2f} 分钟")
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} 执行失败！错误代码: {e.returncode}")
        sys.exit(1)  # 停止后续流程


if __name__ == "__main__":
    print("🌟 开始 2020-2023 年最后四个月数据处理及反演工作流 🌟")
    print("=" * 60)

    # 使用 tqdm 显示大流程进度
    for step in tqdm(WORKFLOW, desc="整套流程进度"):
        run_step(step["name"], step["path"])

    print("\n🎉 所有任务已圆满完成！请查看最终反演图输出目录。")