import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 配置中文字体防乱码 (针对 Windows)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取跑完的 GTWR 结果表
input_csv = r"E:\01Output\Experiments_new\GTWR_Results_Monthly.csv"
save_dir = r"E:\01Output\Experiments_new\GTWR_Plots"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(input_csv)

# 为了画空间图，我们计算每个站点 4 年来的“平均系数值”
spatial_df = df.groupby(['StationCode', 'Longitude', 'Latitude']).mean().reset_index()

# ================= 1. 画 局部 R2 空间分布图 =================
fig, ax = plt.subplots(figsize=(10, 8))
# 用散点图表示空间地图 (颜色越深 R2 越高)
sc = ax.scatter(spatial_df['Longitude'], spatial_df['Latitude'],
                c=spatial_df['Local_R2'], cmap='viridis', s=100, alpha=0.8, edgecolors='k')
plt.colorbar(sc, label='Local $R^2$')
ax.set_title("GTWR 模型局部 $R^2$ 空间分布", fontsize=16)
ax.set_xlabel("经度")
ax.set_ylabel("纬度")
ax.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(save_dir, "GTWR_Local_R2_Map.png"), dpi=300)
plt.close()

# ================= 2. 画 AOD 影响系数空间分布图 =================
fig, ax = plt.subplots(figsize=(10, 8))
# 使用 coolwarm 颜色映射：红色代表正影响，蓝色代表负影响
# 过滤掉不显著的点 (|t_value| < 1.96) 或者对它们做半透明处理
significant_mask = np.abs(spatial_df['t_value_AOD']) > 1.96
sc = ax.scatter(spatial_df.loc[significant_mask, 'Longitude'],
                spatial_df.loc[significant_mask, 'Latitude'],
                c=spatial_df.loc[significant_mask, 'Coef_AOD'],
                cmap='coolwarm', s=100, alpha=0.9, edgecolors='k',
                vmin=-max(abs(spatial_df['Coef_AOD'])), vmax=max(abs(spatial_df['Coef_AOD'])))

# 不显著的点画成灰色小圆圈
ax.scatter(spatial_df.loc[~significant_mask, 'Longitude'],
           spatial_df.loc[~significant_mask, 'Latitude'],
           color='lightgray', s=30, alpha=0.5, label='不显著 (p>0.05)')

plt.colorbar(sc, label='AOD 标准化回归系数')
ax.set_title("AOD 对 PM2.5 影响的空间异质性", fontsize=16)
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(save_dir, "GTWR_AOD_Coef_Map.png"), dpi=300)
plt.close()

# ================= 3. 画 温度(t2m)系数的时间演变折线图 =================
# 按月统计全区的平均系数
temporal_df = df.groupby(['Year', 'Month'])['Coef_t2m'].mean().reset_index()
# 拼接时间字符串
temporal_df['Date'] = pd.to_datetime(temporal_df[['Year', 'Month']].assign(DAY=1))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(temporal_df['Date'], temporal_df['Coef_t2m'], marker='o', linestyle='-', color='coral', linewidth=2)
ax.axhline(0, color='black', linestyle='--', linewidth=1) # 画一条 0 刻度的基准线
ax.set_title("温度 (t2m) 对 PM2.5 影响的时序演变特征", fontsize=16)
ax.set_ylabel("t2m 标准化回归系数")
ax.set_xlabel("时间 (年月)")
ax.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(save_dir, "GTWR_t2m_Temporal_Trend.png"), dpi=300)
plt.close()

print(f"🎨 三张核心图表生成完毕！请去 {save_dir} 文件夹查看。")