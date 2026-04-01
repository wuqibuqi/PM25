import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd

# --- 1. 路径配置 ---
INPUT_TIF = r"E:\Static_Features\Population\CHN_Pop_2020_100m.tif"
OUTPUT_TIF = r"E:\Static_Features\Population\ROI_Pop_2020_100m.tif"

# 你的研究区范围 [北, 南, 东, 西]
NORTH, SOUTH, EAST, WEST = 36, 27, 123, 114


def crop_population_data():
    print("✂️ 正在裁剪人口数据...")

    # 创建裁剪用的矩形框
    roi_geom = [box(WEST, SOUTH, EAST, NORTH)]

    with rasterio.open(INPUT_TIF) as src:
        # 核心操作：只读取 ROI 范围内的像素
        out_image, out_transform = mask(src, roi_geom, crop=True)
        out_meta = src.meta.copy()

        # 更新元数据
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw"  # 使用 LZW 压缩，体积会缩减 10 倍以上
        })

        with rasterio.open(OUTPUT_TIF, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"✅ 裁剪完成！研究区人口数据已保存至: {OUTPUT_TIF}")


if __name__ == "__main__":
    crop_population_data()