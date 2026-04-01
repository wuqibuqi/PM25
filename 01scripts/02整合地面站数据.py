# 把地面站点数据清洗
import pandas as pd
import os
import glob
from tqdm import tqdm

# --- 1. 基础配置 ---
DATA_DIR = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\data_ground"#地面数据xlsx存放路径
OUTPUT_DIR =r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\Output"#清洗数据存放路径

STATION_INFO = {
    "ChunAn": {"name": "淳安", "lon": 119.04, "lat": 29.61},
    "FuYang": {"name": "富阳", "lon": 119.95, "lat": 30.05},
    "JianDe": {"name": "建德", "lon": 119.28, "lat": 29.47},
    "LinAn": {"name": "临安", "lon": 119.72, "lat": 30.23},
    "TongLu": {"name": "桐庐", "lon": 119.68, "lat": 29.79},
    "XiaoShan": {"name": "萧山", "lon": 120.26, "lat": 30.16},
    "YuHang": {"name": "余杭", "lon": 120.30, "lat": 30.42}
}

CN_TO_EN = {v["name"]: k for k, v in STATION_INFO.items()}


def merge_ground_truth():
    all_pm25 = []
    all_met = []

    print("🔍 正在扫描和穿透 Excel 文件...")

    # --- 阶段一：处理 PM2.5 数据 ---
    pm_file = os.path.join(DATA_DIR, "20-24PM.xlsx")
    if os.path.exists(pm_file):
        print(f"📄 正在读取 PM2.5 总表: {os.path.basename(pm_file)}")
        pm_sheets = pd.read_excel(pm_file, sheet_name=None)

        for sheet_name, df_pm in pm_sheets.items():
            if 'PM25_5030' not in df_pm.columns:
                continue

            # 加入 .copy() 消除 Pandas 的强迫症警告
            df_pm = df_pm.dropna(subset=['PM25_5030']).copy()
            df_pm['RealTime'] = pd.to_datetime(df_pm['RealTime'])

            if 'District' not in df_pm.columns:
                df_pm['District'] = sheet_name

            if 'StationNum' in df_pm.columns and 'StationCode' not in df_pm.columns:
                df_pm.rename(columns={'StationNum': 'StationCode'}, inplace=True)
            all_pm25.append(df_pm)
    else:
        print(f"❌ 找不到 {pm_file} ！请检查文件名或路径。")
        return

    # --- 阶段二：处理 气象 数据 (穿透所有 Sheet) ---
    print(f"🌪️ 开始穿透解析气象站点 Excel 的所有 Sheet...")
    met_files = [f for f in glob.glob(os.path.join(DATA_DIR, "*.xlsx")) if "20-24PM" not in os.path.basename(f)]

    for file in tqdm(met_files, desc="气象数据提取进度"):
        file_name = os.path.basename(file)
        district_en = file_name.split('.')[0]

        if district_en in STATION_INFO:
            district_cn = STATION_INFO[district_en]['name']
            met_sheets = pd.read_excel(file, sheet_name=None)

            for sheet_name, df_met in met_sheets.items():
                if 'ObservTimes' not in df_met.columns:
                    continue

                df_met['ObservTimesStr'] = df_met['ObservTimes'].astype(str).str.replace(r'\.0$', '',
                                                                                         regex=True).str.zfill(8)
                df_met['RealTime'] = pd.to_datetime(df_met['ObservTimesStr'], format='%y%m%d%H', errors='coerce')

                # 加入 .copy() 消除警告
                df_met = df_met.dropna(subset=['RealTime']).copy()
                df_met['District'] = district_cn

                if 'StationNum' in df_met.columns and 'StationCode' not in df_met.columns:
                    df_met.rename(columns={'StationNum': 'StationCode'}, inplace=True)

                all_met.append(df_met)

    # --- 阶段三：合并与双键对齐 ---
    print("\n🧬 正在执行时空双键对齐 (地区 + 时间)...")
    pm25_full = pd.concat(all_pm25, ignore_index=True)
    met_full = pd.concat(all_met, ignore_index=True)

    merged_df = pd.merge(pm25_full, met_full, on=['District', 'RealTime'], how='inner')

    # --- 阶段四：空间特征、UTC 时间与严格的时间截断 ---
    print("🌍 正在注入空间特征并裁剪 2020-2023 时间范围...")

    if 'StationCode_x' in merged_df.columns:
        merged_df['StationCode'] = merged_df['StationCode_x']
    elif 'StationCode' not in merged_df.columns:
        merged_df['StationCode'] = "Unknown"

    merged_df['Longitude'] = merged_df['District'].map(
        lambda x: STATION_INFO.get(CN_TO_EN.get(x, ""), {}).get('lon', None))
    merged_df['Latitude'] = merged_df['District'].map(
        lambda x: STATION_INFO.get(CN_TO_EN.get(x, ""), {}).get('lat', None))

    merged_df['UTC_Time'] = merged_df['RealTime'] - pd.Timedelta(hours=8)

    # 【核心新增】：严格截断！只保留到 2023 年 12 月 31 日的数据
    merged_df = merged_df[merged_df['RealTime'].dt.year <= 2023]

    final_cols = [
        'District', 'StationCode', 'Longitude', 'Latitude',
        'RealTime', 'UTC_Time', 'PM25_5030',
        'WindDirect10', 'WindVelocity10', 'DryBulTemp', 'RelHumidity', 'StationPress', 'Precipitation'
    ]

    final_cols = [col for col in final_cols if col in merged_df.columns]
    final_df = merged_df[final_cols].sort_values(by=['RealTime', 'District'])

    # --- 阶段五：完美双输出 (大表 + 各站点小表) ---
    print("💾 正在导出数据...")

    # 1. 导出供深度学习模型训练的总表 (加入 utf-8-sig 彻底解决中文乱码)
    master_output = os.path.join(OUTPUT_DIR, "Ground_Value_2020_2023.csv")#地面真值表
    final_df.to_csv(master_output, index=False, encoding='utf-8-sig')
    print(f"✅ 模型总表已生成: {master_output}")

    # 2. 按照站点单独导出，方便人工画图查看
    station_dir = os.path.join(OUTPUT_DIR, "Station_Split_Data")
    os.makedirs(station_dir, exist_ok=True)

    for district_cn, group_df in final_df.groupby('District'):
        station_path = os.path.join(station_dir, f"{district_cn}_2020_2023.csv")
        group_df.to_csv(station_path, index=False, encoding='utf-8-sig')

    print(f"✅ 各站点独立数据已存入: {station_dir}")
    print(f"📊 总计获得 {len(final_df)} 条 2020-2023 年有效对齐数据！")


if __name__ == "__main__":
    merge_ground_truth()