import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
from scipy.stats import gaussian_kde

target = "PM25_5030"
# 特征定义
features_A = [
    "Longitude", "Latitude", "DOY", "hour",
    "DEM_5km", "Pop_5km", "Roads_5km",
    "CLCD_1_Cropland_Fraction_5km", "CLCD_2_Forest_Fraction_5km",
    "CLCD_5_Water_Fraction_5km", "CLCD_8_Impervious_Fraction_5km",
    "blh", "rh", "t2m", "sp", "u10", "v10", "NDVI", "AOD",
    # "Noise_Test"
]
features_B = [f for f in features_A if f != "AOD"]


def preprocess_data(path):
    print(f"📖 正在读取数据并动态提取特征: {os.path.basename(path)}")
    df = pd.read_csv(path, low_memory=False)
    # 时间特征提取
    times = pd.to_datetime(df['RealTime'])
    df['DOY'] = times.dt.dayofyear
    df['hour'] = times.dt.hour

    # 【新增】：强行注入随机噪点列！
    # 这列数据完全是瞎编的，和 PM2.5 没有任何物理关系
    # df['Noise_Test'] = np.random.randint(0, 2, size=len(df))
    # 生成 0 到 100 之间的随机小数（比如 23.45, 87.12）
    # df['Noise_Test'] = np.random.uniform(0, 100, size=len(df))
    # 生成标准正态分布的随机数
    # df['Noise_Test'] = np.random.randn(len(df))

    # 物理范围限值
    if 'rh' in df.columns: df['rh'] = df['rh'].clip(0, 100)
    return df.dropna(subset=[target])


def draw_single_density_scatter(ax, y_test, y_pred, title):
    """绘制带密度的散点图"""
    sample_size = min(len(y_test), 5000)
    idx = np.random.choice(len(y_test), sample_size, replace=False)
    xs, ys = y_test.values[idx], y_pred[idx]

    xy = np.vstack([xs, ys])
    z = gaussian_kde(xy)(xy)

    sc = ax.scatter(xs, ys, c=z, s=15, cmap='turbo', alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Density')

    # 指标计算
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    lims = [0, max(xs.max(), ys.max())]
    ax.plot(lims, lims, 'k--', lw=1.5, label='1:1 Line')
    m, b = np.polyfit(y_test, y_pred, 1)
    ax.plot(y_test, m * y_test + b, 'r', label=f'Fit: R²={r2:.3f}')

    ax.set_title(f"{title}\n$R^2$={r2:.3f}, RMSE={rmse:.2f}")
    ax.set_xlabel("Observed PM2.5 ($\mu g/m^3$)")
    ax.set_ylabel("Predicted PM2.5 ($\mu g/m^3$)")
    ax.legend()


def run_analysis(data_path, save_dir):
    # --- 动态获取当前处理的文件名前缀 (例如: Train_Data_Exp2_Time) ---
    file_prefix = os.path.splitext(os.path.basename(data_path))[0]

    df_all = preprocess_data(data_path)

    # --- 训练轨道 A (晴天) ---
    print("☀️ 训练 Clear-Sky Model (AOD)...")
    df_A = df_all.dropna(subset=['AOD'])
    X_A = df_A[features_A]
    y_A = df_A[target]
    XA_train, XA_test, yA_train, yA_test = train_test_split(X_A, y_A, test_size=0.2, random_state=42)
    rf_A = RandomForestRegressor(n_estimators=500, max_features='sqrt', n_jobs=-1, random_state=42)
    rf_A.fit(XA_train, yA_train)
    yA_pred = rf_A.predict(XA_test)

    # --- 训练轨道 B (全天候) ---
    print("☁️ 训练 All-Weather Model (Meteo)...")
    X_B = df_all[features_B]
    y_B = df_all[target]
    XB_train, XB_test, yB_train, yB_test = train_test_split(X_B, y_B, test_size=0.2, random_state=42)
    rf_B = RandomForestRegressor(n_estimators=500, max_features='sqrt', n_jobs=-1, random_state=42)
    rf_B.fit(XB_train, yB_train)
    yB_pred = rf_B.predict(XB_test)

    # --- 保存模型 (动态命名) ---
    joblib.dump(rf_A, os.path.join(save_dir, f"{file_prefix}_rf_AOD_model.pkl"))
    joblib.dump(rf_B, os.path.join(save_dir, f"{file_prefix}_rf_Meteo_model.pkl"))

    # --- 产出图片 ---
    print("🎨 正在生成组合图与独立图...")
    sns.set_theme(style="whitegrid")

    # 1. 四合一组合图
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # [0, 0] 轨道 A 重要性
    imp_A = pd.DataFrame({'f': features_A, 'i': rf_A.feature_importances_}).sort_values('i', ascending=False)
    sns.barplot(ax=axes[0, 0], x='i', y='f', data=imp_A, hue='f', palette='Blues_r', legend=False)
    axes[0, 0].set_title(f"1. Feature Importance (Clear-Sky Rail)\n[{file_prefix}]")

    # [0, 1] 轨道 B 重要性
    imp_B = pd.DataFrame({'f': features_B, 'i': rf_B.feature_importances_}).sort_values('i', ascending=False)
    sns.barplot(ax=axes[0, 1], x='i', y='f', data=imp_B, hue='f', palette='Oranges_r', legend=False)
    axes[0, 1].set_title(f"2. Feature Importance (All-Weather Rail)\n[{file_prefix}]")

    # [1, 0] 轨道 A 散点图
    draw_single_density_scatter(axes[1, 0], yA_test, yA_pred, "3. Validation (AOD-Based)")

    # [1, 1] 轨道 B 散点图
    draw_single_density_scatter(axes[1, 1], yB_test, yB_pred, "4. Validation (Meteorology-Based)")

    plt.tight_layout()
    # 动态命名组合图
    combined_img_path = os.path.join(save_dir, f"{file_prefix}_Combined_Dual_Rail_Comparison.png")
    plt.savefig(combined_img_path, dpi=300)

    # 2. 单独生成两张散点图 (动态命名)
    for name, yt, yp in [("Clear_Sky_Scatter", yA_test, yA_pred), ("All_Weather_Scatter", yB_test, yB_pred)]:
        f, ax = plt.subplots(figsize=(8, 7))
        draw_single_density_scatter(ax, yt, yp, name.replace("_", " "))
        single_img_path = os.path.join(save_dir, f"{file_prefix}_{name}.png")
        f.savefig(single_img_path, dpi=300)
        plt.close(f)

    print(f"🎉 {file_prefix} 全部产出！图片保存在: {save_dir}\n")


if __name__ == "__main__":
    # 定义四个实验文件
    exp_files = [
        # "RF_Train_Data_Final.csv",
        "Train_Data_Exp2_Time.csv",
        "Train_Data_Exp3_Space.csv",
        "Train_Data_Mix_PrioTime.csv",
        "Train_Data_Mix_PrioSpace.csv"
    ]

    base_path = r"E:\01Output\Experiments_new"#改路径存训练模型的地方

    for file_name in exp_files:
        print(f"🚀 ===============================================")
        print(f"🚀 正在启动实验训练: {file_name}")

        # 拼接当前文件的完整路径
        current_data_path = os.path.join(base_path, file_name)

        # 为每个实验创建单独的输出文件夹
        current_save_dir = os.path.join(base_path, "Model_Output", "RF", file_name.replace(".csv", ""))
        os.makedirs(current_save_dir, exist_ok=True)

        # 执行分析，将路径作为参数传给函数
        run_analysis(current_data_path, current_save_dir)