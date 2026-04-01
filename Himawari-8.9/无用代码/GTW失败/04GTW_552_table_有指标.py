import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score  # 【新增】引入 R2 计算工具
from mgtwr.model import GTWR

# 1. 路径配置 (请确保路径与你电脑一致)
input_csv = r"E:\01Output\Experiments_new\GTWR_Monthly_Data.csv"
output_csv = r"E:\01Output\Experiments_new\GTWR_Results_Monthly.csv"

print("📖 正在加载 GTWR 月度聚合数据...")
df = pd.read_csv(input_csv)

# 2. 定义变量
target = "PM25_5030"
features = ["AOD", "blh", "t2m", "u10"]

# 3. 提取坐标与时间矩阵
coords = df[['Longitude', 'Latitude']].values
t = df['Time_Index'].values.reshape(-1, 1)

# 4. 提取自变量与因变量
y = df[target].values.reshape(-1, 1)
X = df[features].values

# 标准化处理 (必须做，用于比较谁是主导因子)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ================== 核心提速：跳过寻优，直接写死你的心血参数 ==================
bw_full = 53.20
tau = 3.8000

print(f"⚡ 极速模式启动：直接使用已知最优参数 -> BW: {bw_full}, Tau: {tau}")
print("🚀 正在拟合 13225 行全量 GTWR 模型 (由于不用寻优，这一步几秒钟就能完成)...")

# 拟合模型
model = GTWR(coords, t, X_scaled, y_scaled, bw_full, tau, kernel='gaussian', fixed=False)
results = model.fit()

print(f"📊 整体模型拟合优度 (Global R2): {results.R2:.4f}")

# ================== 提取回归系数 (Betas) 和 t检验值 ==================
print("🔧 正在提取各因子的时空回归系数...")
df['Coef_Intercept'] = results.betas[:, 0]
for i, col in enumerate(features):
    df[f'Coef_{col}'] = results.betas[:, i + 1]
    df[f't_value_{col}'] = results.tvalues[:, i + 1]

# ================== 【核心突破】：手动重构计算 Local_R2 ==================
print("🧮 正在利用数学公式重构预测值，并计算站点级 Local R²...")

# 1. 重构预测值: y_pred = Intercept + (beta1 * X1) + (beta2 * X2) ...
y_pred_scaled = df['Coef_Intercept'].values.copy()
for i, col in enumerate(features):
    y_pred_scaled += df[f'Coef_{col}'].values * X_scaled[:, i]

# 把缩放后的真实值和预测值存入临时列
df['y_true_scaled'] = y_scaled.flatten()
df['y_pred_scaled'] = y_pred_scaled

# 2. 按站点分组计算 R²
station_r2_dict = {}
for station, group in df.groupby('StationCode'):
    if len(group) > 3:  # 至少有几个月的数据才算 R²，保证统计学意义
        r2 = r2_score(group['y_true_scaled'], group['y_pred_scaled'])
        # 限制在 0~1 之间，防止个别极差站点出现负数导致地图颜色突变
        station_r2_dict[station] = max(0.0, min(1.0, r2))
    else:
        station_r2_dict[station] = np.nan

# 3. 把算好的 R² 映射回大表 (每一个站点对应的所有行，都会附上该站点的 Local_R2)
df['Local_R2'] = df['StationCode'].map(station_r2_dict)

print(f"✅ Local R² 计算完毕！全区平均站点 Local R²: {df['Local_R2'].mean():.4f}")

# 清理不需要输出的临时列
df.drop(columns=['y_true_scaled', 'y_pred_scaled'], inplace=True)

# ================== 写入硬盘 ==================
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"🎉 完美解决！包含全部系数和 Local_R2 的史诗级 GTWR 结果已成功保存至: {output_csv}")