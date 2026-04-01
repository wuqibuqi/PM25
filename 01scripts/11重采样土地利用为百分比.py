import os
import rioxarray
import xarray as xr
import numpy as np
from rasterio.enums import Resampling

# --- 1. 路径配置 ---
CLCD_30M_PATH = r"E:\CLCD_Data\CLCD_WGS84\CLCD_ROI_2020.tif"
MASTER_TIF = r"E:\Himawari-8_TIFF\202001\01\H08_20200101_0000_1HARP031_FLDK.02401_02401.tif"
OUTPUT_DIR = r"E:\Standard_Dataset_5km\Static"

os.makedirs(OUTPUT_DIR, exist_ok=True)

clcd_classes = {
    1: "Cropland", 2: "Forest", 3: "Shrub", 4: "Grassland",
    5: "Water", 6: "Snow_Ice", 7: "Barren", 8: "Impervious", 9: "Wetland"
}


def generate_fractional_maps_fixed():
    print("📖 正在加载 181x181 大师模板...")
    master = rioxarray.open_rasterio(MASTER_TIF)

    print("📖 正在加载 30m 高精度 CLCD 原图...")
    clcd_da = rioxarray.open_rasterio(CLCD_30M_PATH)

    # 提取底层 numpy 矩阵和原始空值标签
    data_np = clcd_da.values
    orig_nodata = clcd_da.rio.nodata

    for class_id, class_name in clcd_classes.items():
        out_path = os.path.join(OUTPUT_DIR, f"CLCD_{class_id}_{class_name}_Fraction_5km.tif")
        print(f"⏳ 正在精确计算 [{class_name}] 的 5km 面积占比...")

        # 1. 初始化全 0.0 矩阵 (这意味着默认全是 0% 占比)
        target_np = np.zeros_like(data_np, dtype="float32")

        # 2. 把属于该类的像素设为 1.0
        target_np[data_np == class_id] = 1.0

        # 3. 核心修复：把真正的原始空值（比如海面）设为 NaN，而不是 0！
        if orig_nodata is not None:
            target_np[data_np == orig_nodata] = np.nan

        # 4. 转回 DataArray 并严格声明 NaN 才是空值
        binary_da = xr.DataArray(target_np, coords=clcd_da.coords, dims=clcd_da.dims)
        binary_da.rio.write_crs(clcd_da.rio.crs, inplace=True)
        binary_da.rio.write_transform(clcd_da.rio.transform(), inplace=True)
        binary_da.rio.write_nodata(np.nan, inplace=True)  # 灵魂代码！

        # 5. 重采样 (此时 0.0 会被正确视为数值参与平均，只有 NaN 被忽略)
        fractional_map = binary_da.rio.reproject_match(master, resampling=Resampling.average)

        # 6. 强制覆盖保存
        fractional_map.rio.to_raster(out_path, compress="lzw")
        print(f"✅ 成功覆盖: {class_name}")

    print("\n🎉 真正无暇的【高精度土地利用占比图】生成完毕！")


if __name__ == "__main__":
    generate_fractional_maps_fixed()