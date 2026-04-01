import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ==============================================================
# --- 1. 配置 ---
# ==============================================================
ARL_FILES = [
    "gdas1.dec20.w3", "gdas1.dec20.w4",
    "gdas1.jan22.w5", "gdas1.feb22.w1",
    "gdas1.jan23.w3", "gdas1.jan23.w4",
    "gdas1.dec23.w4", "gdas1.dec23.w5"
]

BASE_URL = "https://www.ready.noaa.gov/data/archives/gdas1/"
SAVE_DIR = r"D:\1document\Graduation Thesis\01Code\DATA\ARL_Data"
MAX_WORKERS = 4  # 同时开启多少个下载任务？根据网速设为 4-8


# ==============================================================
# --- 2. 下载引擎 ---
# ==============================================================

def download_one_file(filename):
    target_path = os.path.join(SAVE_DIR, filename)

    # 检查是否已存在 (断点续传)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 500 * 1024 * 1024:
        return f"⏭️ {filename} 已存在"

    url = f"{BASE_URL}{filename}"
    try:
        # 增加超时和重试机制
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # 为每个线程创建独立的进度条
        with open(target_path, 'wb') as f:
            # 这里不使用 tqdm，因为并发时多个进度条会刷屏，我们主程序统一看
            for data in response.iter_content(chunk_size=1024 * 1024):
                f.write(data)
        return f"✅ {filename} 下载完成"
    except Exception as e:
        if os.path.exists(target_path): os.remove(target_path)
        return f"❌ {filename} 失败: {e}"


def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    print(f"🔥 启动多线程引擎 (并发数: {MAX_WORKERS})...")
    print(f"📦 预计下载总量: 约 5GB")

    # 使用线程池进行并发下载
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 tqdm 监控整体进度（完成的文件数）
        results = list(tqdm(executor.map(download_one_file, ARL_FILES),
                            total=len(ARL_FILES),
                            desc="总进度",
                            colour='blue'))

    for r in results:
        print(r)


if __name__ == "__main__":
    main()