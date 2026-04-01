import os
import glob
import pandas as pd
from tqdm import tqdm

# ================= 1. 配置参数 =================
DATA_ROOT = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\AIR"
OLD_GROUND_CSV = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\Output\Ground_Value_2020_2023.csv"
NEW_GROUND_CSV = r"D:\1document\Graduation Thesis\01Code\Himawari-8.9\Data\Output\Ground_Value_2020_2023_More.csv"

# 【关键配置】研究区经纬度边界 (目前设定为近似浙江省的范围)
MIN_LON = 114.0
MAX_LON = 123.0
MIN_LAT = 27.0
MAX_LAT = 36.0

# 你的 PM2.5 目标列名
TARGET_COL = "PM25_5030"

# ================= 2. 动态读取并整合所有站点列表 =================
print("📍 正在扫描并整合历史站点列表...")
station_files = glob.glob(os.path.join(DATA_ROOT, "站点列表*.csv"))

if not station_files:
    print("❌ 错误：未在目录下找到任何 '站点列表' 开头的 CSV 文件，请检查路径。")
    exit()

all_stations = []
for sf in station_files:
    try:
        df = pd.read_csv(sf, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(sf, encoding='gbk')
    all_stations.append(df)

stations_df = pd.concat(all_stations, ignore_index=True)

# 【修正：完美对齐原表列名】
stations_df.rename(columns={
    '监测点编码': 'StationCode',
    '监测点名称': 'District',
    '经度': 'Longitude',
    '纬度': 'Latitude'
}, inplace=True)

stations_df = stations_df.drop_duplicates(subset=['StationCode'], keep='last')

# 强制清洗经纬度列
stations_df['Longitude'] = pd.to_numeric(stations_df['Longitude'], errors='coerce')
stations_df['Latitude'] = pd.to_numeric(stations_df['Latitude'], errors='coerce')
stations_df = stations_df.dropna(subset=['Longitude', 'Latitude'])

# 筛选研究区边界
target_stations = stations_df[
    (stations_df['Longitude'] >= MIN_LON) &
    (stations_df['Longitude'] <= MAX_LON) &
    (stations_df['Latitude'] >= MIN_LAT) &
    (stations_df['Latitude'] <= MAX_LAT)
    ].copy()

# 建立字典映射
site_info = target_stations.set_index('StationCode')[['District', 'Longitude', 'Latitude']].to_dict('index')
valid_site_codes = list(site_info.keys())

print(f"✅ 成功整合 {len(station_files)} 份站点文件。在研究区内共锁定 {len(valid_site_codes)} 个有效监测点。")

# ================= 3. 遍历每日数据提取 PM2.5 =================
print("⏳ 正在遍历 2020-2023 年每日空气质量文件...")

site_files = glob.glob(os.path.join(DATA_ROOT, "**", "china_sites_*.csv"), recursive=True)
all_extracted_data = []

for file_path in tqdm(site_files, desc="处理每日数据"):
    try:
        df_daily = pd.read_csv(file_path, encoding='utf-8', low_memory=False)

        df_pm25 = df_daily[df_daily['type'] == 'PM2.5'].copy()
        if df_pm25.empty: continue

        existing_target_cols = [col for col in valid_site_codes if col in df_pm25.columns]
        if not existing_target_cols: continue

        keep_cols = ['date', 'hour'] + existing_target_cols
        df_pm25 = df_pm25[keep_cols]

        df_melted = df_pm25.melt(id_vars=['date', 'hour'],
                                 value_vars=existing_target_cols,
                                 var_name='StationCode',
                                 value_name=TARGET_COL)

        df_melted[TARGET_COL] = pd.to_numeric(df_melted[TARGET_COL], errors='coerce')
        df_melted = df_melted.dropna(subset=[TARGET_COL])

        # 时间处理
        df_melted['date'] = df_melted['date'].astype(str)
        df_melted['RealTime'] = pd.to_datetime(df_melted['date'], format='%Y%m%d') + \
                                pd.to_timedelta(df_melted['hour'], unit='h')
        df_melted['UTC_Time'] = df_melted['RealTime'] - pd.Timedelta(hours=8)

        # 映射字段
        df_melted['District'] = df_melted['StationCode'].map(lambda x: site_info[x]['District'])
        df_melted['Longitude'] = df_melted['StationCode'].map(lambda x: site_info[x]['Longitude'])
        df_melted['Latitude'] = df_melted['StationCode'].map(lambda x: site_info[x]['Latitude'])

        final_daily = df_melted[
            ['StationCode', 'District', 'RealTime', 'UTC_Time', 'Longitude', 'Latitude', TARGET_COL]]
        all_extracted_data.append(final_daily)

    except Exception as e:
        print(f"\n❌ 读取文件出错 {os.path.basename(file_path)}: {e}")

# ================= 4. 与原数据合并并输出 =================
if not all_extracted_data:
    print("❌ 未提取到任何符合条件的数据，请检查经纬度范围和原始数据内容。")
    exit()

new_data_df = pd.concat(all_extracted_data, ignore_index=True)
print(f"✨ 成功从开源数据集中提取了 {len(new_data_df)} 条国控点 PM2.5 记录。")

print("🔄 正在与原有的 Ground_Value 数据合并...")
if os.path.exists(OLD_GROUND_CSV):
    old_df = pd.read_csv(OLD_GROUND_CSV)
    old_df['RealTime'] = pd.to_datetime(old_df['RealTime'])

    if 'UTC_Time' not in old_df.columns:
        old_df['UTC_Time'] = old_df['RealTime'] - pd.Timedelta(hours=8)

    combined_df = pd.concat([old_df, new_data_df], ignore_index=True)

    combined_df['Longitude_round'] = combined_df['Longitude'].round(4)
    combined_df['Latitude_round'] = combined_df['Latitude'].round(4)
    combined_df = combined_df.drop_duplicates(subset=['RealTime', 'Longitude_round', 'Latitude_round'])
    combined_df = combined_df.drop(columns=['Longitude_round', 'Latitude_round'])

else:
    print("⚠️ 未找到旧的 Ground_Value 文件，将直接保存新提取的数据。")
    combined_df = new_data_df

# 统一列顺序
cols_order = ['StationCode', 'District', 'RealTime', 'UTC_Time', 'Longitude', 'Latitude', TARGET_COL]
other_cols = [c for c in combined_df.columns if c not in cols_order]
combined_df = combined_df[cols_order + other_cols]

combined_df.to_csv(NEW_GROUND_CSV, index=False, encoding='utf-8-sig')
print(f"🎉 全部完成！新的扩充站点文件已保存至: {NEW_GROUND_CSV}")