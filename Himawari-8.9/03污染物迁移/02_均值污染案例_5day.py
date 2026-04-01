import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 基础配置 ---
CSV_PATH = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\Hangzhou_Hourly_Mean.csv")
OUTPUT_DIR = Path(r"D:\1document\Graduation Thesis\01Code\DATA\PM2.5_Meteorology_Study\PM2.5_Meteorology_Study_5day")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def find_refined_golden_events(df):
    print("\n🔍 正在进行高精度气象诊断（已调整为 5-6 天完整周期）...")

    # 标注跨年冬季年份
    df['Winter_Year'] = df.index.year
    jan_feb_mask = df.index.month.isin([1, 2])
    df.loc[jan_feb_mask, 'Winter_Year'] = df.index[jan_feb_mask].year - 1

    results = []
    winter_mask = df.index.month.isin([11, 12, 1, 2])
    winter_df = df[winter_mask].copy()

    for year in range(2020, 2024):
        year_data = winter_df[winter_df['Winter_Year'] == year].copy()
        if year_data.empty: continue

        # 寻找爆发力最强时刻 (12小时增量)
        year_data['diff_12h'] = year_data['HZ_Mean'].diff(periods=12)
        burst_time = year_data['diff_12h'].idxmax()

        # 寻找随后的峰值时刻
        search_window = year_data.loc[burst_time: burst_time + pd.Timedelta(hours=48)]
        if search_window.empty: continue
        peak_time = search_window['HZ_Mean'].idxmax()
        max_val = search_window['HZ_Mean'].max()

        # 【核心修改】：统一设定为 5.5 天的展示窗口
        # 从爆发前 48 小时开始（看到完整的清洁背景）
        # 到峰值后 72 小时结束（看到完整的消散过程）
        start_show = burst_time - pd.Timedelta(days=2)
        end_show = peak_time + pd.Timedelta(days=3)

        results.append({
            '冬季案例': f"{year}-{year + 1} Winter",
            '爆发时刻': burst_time.strftime('%Y-%m-%d %H:00'),
            '峰值时刻': peak_time.strftime('%Y-%m-%d %H:00'),
            '最高浓度': round(max_val, 2),
            '分析开始': start_show.strftime('%Y-%m-%d %H:00'),
            '分析结束': end_show.strftime('%Y-%m-%d %H:00'),
            '持续天数': round((end_show - start_show).total_seconds() / 86400, 1)
        })

    return pd.DataFrame(results)


def main():
    if not CSV_PATH.exists():
        print("❌ 错误：未找到 Hangzhou_Hourly_Mean.csv，请检查路径。")
        return

    df = pd.read_csv(CSV_PATH, index_col='Time', parse_dates=True)
    df = df.sort_index()

    # 执行高精度识别
    cases_df = find_refined_golden_events(df)

    print("\n✨ 重新校准后的四年黄金时间段 ✨")
    print(cases_df.to_string(index=False))

    # 保存结果
    cases_df.to_csv(OUTPUT_DIR / "Refined_Golden_Cases_Selection.csv", index=False, encoding='utf-8-sig')

    # 绘图：检查阴影是否精准覆盖了“大尖峰”
    plt.figure(figsize=(16, 8))
    plt.plot(df.index, df['HZ_Mean'], color='#2c3e50', linewidth=0.5, alpha=0.5, label='PM2.5 均值序列')

    colors = ['#FF4500', '#FFD700', '#32CD32', '#1E90FF']  # 四种颜色代表四年
    for i, (_, row) in enumerate(cases_df.iterrows()):
        plt.axvspan(pd.to_datetime(row['分析开始']), pd.to_datetime(row['分析结束']),
                    color=colors[i], alpha=0.4, label=row['冬季案例'])

        # 标注峰值
        plt.annotate(f"Peak: {row['最高浓度']}",
                     xy=(pd.to_datetime(row['峰值时刻']), row['最高浓度']),
                     xytext=(10, 10), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color=colors[i]),
                     fontsize=9, color=colors[i], fontweight='bold')

    plt.axhline(y=75, color='red', linestyle='--', alpha=0.3, label='二级标准(75)')
    plt.title("长三角 PM2.5 迁移典型案例：四年冬季爆发时段精准锁定", fontsize=16)
    plt.ylabel("浓度 ($\mu g/m^3$)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.2)

    # 局部放大显示（可选）：如果想看清楚 2023 年那个大爆发，可以取消下面注释
    # plt.xlim(pd.to_datetime('2022-12-01'), pd.to_datetime('2023-03-01'))

    plt.savefig(OUTPUT_DIR / "Hangzhou_Timeline_Final_Calibration.png", dpi=300)
    print(f"\n✅ 识别完成！请查看新生成的趋势图，确认阴影是否已经准确锁定了那四个最高的‘山峰’。")


if __name__ == "__main__":
    main()