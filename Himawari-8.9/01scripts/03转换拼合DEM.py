# 转换拼合nasa下载的dem数据
import rasterio
from rasterio.merge import merge
import glob
import os
import zipfile

# --- 1. 路径配置 ---
DEM_DIR = r"E:\DEM"  # nasa数据下载目录
OUTPUT_FILE = os.path.join(DEM_DIR, "ChangSanJiao_DEM_30m.tif")


def extract_and_mosaic():
    # A. 自动解压所有的 zip (如果还没解压的话)
    zip_files = glob.glob(os.path.join(DEM_DIR, "*.zip"))
    print(f"📦 正在检查并解压 {len(zip_files)} 个压缩包...")
    for z in zip_files:
        with zipfile.ZipFile(z, 'r') as zip_ref:
            zip_ref.extractall(DEM_DIR)

    # B. 寻找所有的 .hgt 文件进行缝合
    hgt_files = glob.glob(os.path.join(DEM_DIR, "*.hgt"))
    if not hgt_files:
        print("❌ 没找到解压后的 .hgt 文件，请确认解压是否成功！")
        return

    print(f"🧩 正在缝合 {len(hgt_files)} 个 DEM 碎块...")
    src_files_to_mosaic = [rasterio.open(f) for f in hgt_files]

    # C. 执行合并
    mosaic, out_trans = merge(src_files_to_mosaic)

    # D. 更新元数据
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": src_files_to_mosaic[0].crs,
        "compress": "lzw"  # 压缩一下，否则文件会非常大
    })

    # E. 保存最终大图
    with rasterio.open(OUTPUT_FILE, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"✅ 大功告成！完整 DEM 已保存至: {OUTPUT_FILE}")

    # 关闭资源
    for src in src_files_to_mosaic:
        src.close()


if __name__ == "__main__":
    extract_and_mosaic()