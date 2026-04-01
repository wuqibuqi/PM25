import os
import glob
import numpy as np
import rasterio
from rasterio.plot import plotting_extent
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import ImageGrid
import gc

# ================= 1. 路径与核心配置 =================
# 输入路径：提取后的杭州逐月 TIF 根目录 (07脚本的输出)
INPUT_ROOT = r"D:\1document\Graduation Thesis\01Code\DATA\Hangzhou_Monthly_Analysis"
# 输出路径：最终高清组合图片保存目录
PLOT_SAVE_ROOT = r"D:\1document\Graduation Thesis\01Code\DATA\Final_Paper_Plots\Hangzhou_Final_Report_Maps"

# 设定处理年份
YEARS = ["2020", "2021", "2022", "2023"]

# 🌟 核心需求：色带选择 'coolwarm' (蓝-红渐变)
CMAP_NAME = 'coolwarm'

# 学术字体设置 (确保 D 盘已安装 Arial 字体，或改为 SimHei 并注释 minus 设置)
plt.rcParams['font.sans-serif'] = ['Arial']  # 学术通用 Arial
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

os.makedirs(PLOT_SAVE_ROOT, exist_ok=True)


# ================= 2. 自动化组合绘图引擎 =================

def plot_final_composite_maps():


    print(f"🎬 正在启动年度全景组合绘图任务 (大画幅、英文月份、边界余量)...")
    print(f"🎨 使用色带: {CMAP_NAME}，颜色上限将根据每年数据动态计算。")

    for year in YEARS:
        year_in_path = os.path.join(INPUT_ROOT, year)
        if not os.path.exists(year_in_path):
            print(f"⏩ 跳过 {year} 年 (未找到数据目录)")
            continue

        print(f"📅 正在处理 {year} 年数据...")

        # --- 步骤 A: 预读取全年的 TIF，计算动态 Vmax 与地理缓冲区 ---
        year_data_list = []

        # 🌟 需求 4: 直接用英语写月份，不出现年月
        month_labels = ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December']

        raw_extent = None

        # 尝试读取 12 个月
        for month in range(1, 13):
            mm_str = f"{month:02d}"
            tif_pattern = os.path.join(year_in_path, f"HZ_PM25_MonthlyMean_{year}{mm_str}.tif")
            tif_files = glob.glob(tif_pattern)

            if tif_files:
                tif_path = tif_files[0]
                try:
                    with rasterio.open(tif_path) as src:
                        data = src.read(1).astype(np.float32)
                        # 将 nodata 转为 NaN
                        data[data == src.nodata] = np.nan
                        year_data_list.append(data)
                        # 获取原始地理范围 (假设全年 TIF 投影和范围一致)
                        if raw_extent is None:
                            raw_extent = plotting_extent(src)
                except Exception:
                    print(f"❌ 读取失败: {os.path.basename(tif_path)}")
                    year_data_list.append(None)
            else:
                year_data_list.append(None)

        # 计算年度全局最大值用于动态设定 Vmax
        valid_data_for_max = [d for d in year_data_list if d is not None]
        if not valid_data_for_max:
            continue

        yearly_max = np.nanmax([np.nanmax(d) for d in valid_data_for_max])
        # 向上取整，使颜色条刻度整齐
        vmax_plot = np.ceil(yearly_max / 10) * 10
        print(f"   📈 {year} 年动态颜色上限 (Vmax) 设定为: {vmax_plot:.1f} μg/m3")

        # 🌟 需求 1: 杭州边界留余量 (计算缓冲区)
        # WGS84 坐标系下，增加 10% 的余量
        xmin, xmax, ymin, ymax = raw_extent
        x_range = xmax - xmin
        y_range = ymax - ymin
        buffer_percent = 0.10  # 10% 缓冲区

        buffered_extent = (
            xmin - x_range * buffer_percent,
            xmax + x_range * buffer_percent,
            ymin - y_range * buffer_percent,
            ymax + y_range * buffer_percent
        )

        # --- 步骤 B: 绘制大尺寸画布 ---
        # 保持之前的物理尺寸 (24x18 英寸)，确保高分辨率
        fig = plt.figure(figsize=(24, 18), dpi=300)

        # 创建 ImageGrid 布局: 3行4列，共享颜色条
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(3, 4),  # 3行4列
                         axes_pad=0.5,  # 适当收紧子图间距，让图片占比更大
                         label_mode="L",  # 仅在左侧和底部显示经纬度
                         cbar_mode="single",  # 共享一个颜色条
                         cbar_location="right",
                         cbar_pad=0.5,  # 颜色条与子图的间距
                         cbar_size="2.5%"  # 颜色条宽度占比
                         )

        # 遍历 12 个子图进行绘制
        for i, ax in enumerate(grid):
            # 获取对应月份数据
            month_data = year_data_list[i]
            # 🌟 需求 4: 使用英文月份标题
            month_title = month_labels[i]

            if month_data is not None:
                # 绘制地图底层 (固定 vmin=0, vmax=vmax_plot, 色带 coolwarm)
                im = ax.imshow(month_data, extent=raw_extent, cmap=CMAP_NAME,
                               vmin=0, vmax=70, origin='upper')

                # --- 子图学术修饰 ---
                # 🌟 需求 4: 英文标题，加大加粗，适合大图
                ax.set_title(month_title, fontsize=20, fontweight='bold', pad=10)

                # 🌟 需求 1: 应用带余量的地理范围
                ax.set_xlim(buffered_extent[0], buffered_extent[1])
                ax.set_ylim(buffered_extent[2], buffered_extent[3])

                # 坐标轴格式化 (度分秒)
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f°E'))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f°N'))
                ax.tick_params(labelsize=12)
            else:
                # 缺月处理
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', fontsize=16)
                ax.set_title(month_title, fontsize=20, fontweight='bold')
                ax.set_xticks([]);
                ax.set_yticks([])  # 隐藏空白图刻度

        # --- 3. 共享颜色条设置 (Colorbar) ---
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.set_label('PM$_{2.5}$ Concentration ($\mu g/m^3$)', fontsize=18, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
        # 设定 Colorbar 的刻度间距 (动态，每 10 或 20 一个刻度)
        tick_spacing = 20 if vmax_plot > 90 else 10
        cbar.set_ticks(np.arange(0, 70 + 1, tick_spacing))

        # --- 4. 调整布局与保存 ---
        # 🌟 需求 2 & 3: 取消总标题，整体上移缩紧间距
        # 通过调节 bottom 和 top 参数实现
        plt.subplots_adjust(left=0.08, right=0.88, top=0.98, bottom=0.1)

        # 保存组合大图
        plot_name = f"HZ_Composite_MonthlyMean_{year}.png"
        plot_save_path = os.path.join(PLOT_SAVE_ROOT, plot_name)
        # bbox_inches='tight' 自动剔除多余白边
        plt.savefig(plot_save_path, bbox_inches='tight', dpi=200)
        plt.close(fig)  # 关键：及时关闭画布

        # 清理该年内存
        del year_data_list, valid_data_for_max, fig, grid
        gc.collect()

    print(f"\n✨ 任务圆满完成！大画幅最终学术地图保存于: {PLOT_SAVE_ROOT}")

if __name__ == "__main__":
    plot_final_composite_maps()