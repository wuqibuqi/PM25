import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from mgtwr.sel import SearchGTWRParameter
from mgtwr.model import GTWR

# 1. 路径配置 (请替换为你上一步生成的 GTWR 聚合表路径)
input_csv = r"E:\01Output\Experiments_new\GTWR_Monthly_Data.csv"
output_csv = r"E:\01Output\Experiments_new\GTWR_Results_Monthly.csv"

print("📖 正在加载 GTWR 月度聚合数据...")
df = pd.read_csv(input_csv)

# 2. 定义变量
target = "PM25_5030"
# 【关键】：这里放你随机森林特征重要性排名前 4-5 的核心因子
features = ["AOD", "blh", "t2m", "u10"]

# 3. 提取坐标与时间矩阵
coords = df[['Longitude', 'Latitude']].values
# 时间序列必须转为二维数组
t = df['Time_Index'].values.reshape(-1, 1)

# 4. 提取自变量与因变量 (必须转为二维数组)
y = df[target].values.reshape(-1, 1)
X = df[features].values

# 【强烈建议：数据标准化】
# 为了后续能比较“哪个因子是主导因子”，必须将 X 和 y 标准化 (均值为0，方差为1)
# 这样得出的系数就是“标准化回归系数”
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 5. GTWR 核心运算：寻找最优时空带宽 (Bandwidth)
print("⏳ 正在自动寻优 GTWR 时空带宽 (这可能需要几分钟)...")
# 参数解释：fixed=False 表示使用自适应带宽 (KNN思想，更适合站点分布不均的情况)
sel = SearchGTWRParameter(coords, t, X_scaled, y_scaled, kernel='gaussian', fixed=False)
bw, tau = sel.search()
print(f"✅ 最优空间带宽 (BW): {bw:.2f}")
print(f"✅ 最优时间惩罚参数 (Tau): {tau:.4f}")

# 6. 拟合并拟合 GTWR 模型
print("🚀 正在拟合 GTWR 模型计算时空局部系数...")
model = GTWR(coords, t, X_scaled, y_scaled, bw, tau, kernel='gaussian', fixed=False)
results = model.fit()

# 7. 提取结果并保存
print(f"📊 整体模型拟合优度 (Global R2): {results.R2:.4f}")

# 将结果拼回原数据表
df['Local_R2'] = results.localR2
# 提取截距项 (Intercept) 和 各个变量的系数
df['Coef_Intercept'] = results.betas[:, 0]
for i, col in enumerate(features):
    df[f'Coef_{col}'] = results.betas[:, i + 1] # 索引 0 是截距，特征从 1 开始
    # 提取 t 值用于显著性检验 (绝对值 > 1.96 视为 95% 显著)
    df[f't_value_{col}'] = results.tvalues[:, i + 1]

df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"🎉 GTWR 结果已成功保存至: {output_csv}")