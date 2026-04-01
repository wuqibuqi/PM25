import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
import gc
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
    try:
        df = pd.read_csv(path, engine='pyarrow')
    except:
        df = pd.read_csv(path, low_memory=False)

    times = pd.to_datetime(df['RealTime'])
    df['DOY'] = times.dt.dayofyear
    df['hour'] = times.dt.hour

    if 'rh' in df.columns: df['rh'] = df['rh'].clip(0, 100)
    df = df.dropna(subset=[target])

    # ================= 【新增：地球物理常识大清洗】 =================
    print(f"🧹 清洗前数据量: {len(df)}")

    # 1. PM2.5 门槛：剔除负数和极其离谱的极值 (中国历史极端最高约在一两千左右，设为 2000 封顶)
    df = df[(df[target] >= 0) & (df[target] <= 2000)]

    # 2. AOD 门槛：剔除负数和异常高值 (如果列存在的话)
    if 'AOD' in df.columns:
        # AOD 为 NaN 是允许的(轨道 B 要用)，所以保留 NaN 或者在 0~5 之间的值
        df = df[df['AOD'].isna() | ((df['AOD'] >= 0) & (df['AOD'] <= 5.0))]

    # 3. 气象异常门槛：防止出现 9999 等填充值
    # 边界层高度 (blh) 一般在几十到几千米；温度 (t2m) 是开尔文或摄氏度，不会上万；气压 (sp) 同理
    for col in ['blh', 't2m', 'sp', 'u10', 'v10']:
        if col in df.columns:
            df = df[(df[col] > -1000) & (df[col] < 100000)]  # 设定一个绝对宽容但能防 1e10 的范围

    print(f"✨ 清洗后数据量: {len(df)} (剔除了 {len(df) - len(df[df[target] <= 2000])} 个潜在异常点)")  # 简单统计
    # ==============================================================

    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')

    return df


def draw_single_density_scatter(ax, y_test, y_pred, title):
    sample_size = min(len(y_test), 5000)
    idx = np.random.choice(len(y_test), sample_size, replace=False)
    xs, ys = y_test.values[idx], y_pred[idx]

    xy = np.vstack([xs, ys])
    z = gaussian_kde(xy)(xy)

    sc = ax.scatter(xs, ys, c=z, s=15, cmap='turbo', alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Density')

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
    file_prefix = os.path.splitext(os.path.basename(data_path))[0]
    df_all = preprocess_data(data_path)

    # ==================== 轨道 A 训练 ====================
    print("☀️ 训练 Clear-Sky Model (AOD)...")
    df_A = df_all.dropna(subset=['AOD'])
    X_A = df_A[features_A]
    y_A = df_A[target]

    # 【极致内存清理】：提取出 X 和 Y 后，母表立刻删除，拯救数 GB 内存！
    del df_A
    gc.collect()

    XA_train, XA_test, yA_train, yA_test = train_test_split(X_A, y_A, test_size=0.2, random_state=42)

    # 训练集拆分完毕，原始 X_A 和 y_A 也可以去死了
    del X_A, y_A
    gc.collect()

    # 【模型极限提速】：加入 max_samples=0.2，训练速度飙升 5 倍！
    rf_A = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_samples=0.7,  # 👈 同样应用 20% 子采样率
        max_features=0.8,
        n_jobs=-1,
        random_state=42
    )
    rf_A.fit(XA_train, yA_train)
    yA_pred = rf_A.predict(XA_test)

    # 轨道 A 训练完毕，立刻清空训练集
    del XA_train, XA_test, yA_train
    gc.collect()

    # ==================== 轨道 B 训练 ====================
    print("☁️ 训练 All-Weather Model (Meteo)...")
    X_B = df_all[features_B]
    y_B = df_all[target]

    # 轨道 B 提取完毕，彻底清空全局母表 df_all
    del df_all
    gc.collect()

    XB_train, XB_test, yB_train, yB_test = train_test_split(X_B, y_B, test_size=0.2, random_state=42)

    del X_B, y_B
    gc.collect()

    rf_B = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_samples=0.7,  # 👈 同样应用 20% 子采样率
        max_features=0.8,
        n_jobs=-1,
        random_state=42
    )
    rf_B.fit(XB_train, yB_train)
    yB_pred = rf_B.predict(XB_test)

    del XB_train, XB_test, yB_train
    gc.collect()

    # --- 保存模型 ---
    print("💾 正在保存模型至硬盘...")
    joblib.dump(rf_A, os.path.join(save_dir, f"{file_prefix}_rf_AOD_model.pkl"))
    joblib.dump(rf_B, os.path.join(save_dir, f"{file_prefix}_rf_Meteo_model.pkl"))

    # --- 产出图片 ---
    print("🎨 正在生成组合图与独立图...")
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    imp_A = pd.DataFrame({'f': features_A, 'i': rf_A.feature_importances_}).sort_values('i', ascending=False)
    sns.barplot(ax=axes[0, 0], x='i', y='f', data=imp_A, hue='f', palette='Blues_r', legend=False)
    axes[0, 0].set_title(f"1. Feature Importance (Clear-Sky Rail)\n[{file_prefix}]")

    imp_B = pd.DataFrame({'f': features_B, 'i': rf_B.feature_importances_}).sort_values('i', ascending=False)
    sns.barplot(ax=axes[0, 1], x='i', y='f', data=imp_B, hue='f', palette='Oranges_r', legend=False)
    axes[0, 1].set_title(f"2. Feature Importance (All-Weather Rail)\n[{file_prefix}]")

    draw_single_density_scatter(axes[1, 0], yA_test, yA_pred, "3. Validation (AOD-Based)")
    draw_single_density_scatter(axes[1, 1], yB_test, yB_pred, "4. Validation (Meteorology-Based)")

    plt.tight_layout()
    combined_img_path = os.path.join(save_dir, f"{file_prefix}_Combined_Dual_Rail_Comparison.png")
    plt.savefig(combined_img_path, dpi=300)

    for name, yt, yp in [("Clear_Sky_Scatter", yA_test, yA_pred), ("All_Weather_Scatter", yB_test, yB_pred)]:
        f, ax = plt.subplots(figsize=(8, 7))
        draw_single_density_scatter(ax, yt, yp, name.replace("_", " "))
        single_img_path = os.path.join(save_dir, f"{file_prefix}_{name}.png")
        f.savefig(single_img_path, dpi=300)
        plt.close(f)

    # 循环结束前的最终大扫除
    plt.close('all')
    del rf_A, rf_B, imp_A, imp_B
    gc.collect()

    print(f"🎉 {file_prefix} 全部产出！图片保存在: {save_dir}\n")


if __name__ == "__main__":
    exp_files = [
        "Train_Data_Exp2_Time.csv",
        "Train_Data_Exp3_Space.csv",
        "Train_Data_Mix_PrioTime.csv",
        "Train_Data_Mix_PrioSpace.csv"
    ]

    base_path = r"E:\01Output\Experiments_new"

    for file_name in exp_files:
        print(f"🚀 ===============================================")
        print(f"🚀 正在启动实验训练: {file_name}")

        current_data_path = os.path.join(base_path, file_name)
        if not os.path.exists(current_data_path):
            print(f"⚠️ 文件 {file_name} 不存在，已跳过。")
            continue

        current_save_dir = os.path.join(base_path, "Model_Output", "RF_552", file_name.replace(".csv", ""))
        os.makedirs(current_save_dir, exist_ok=True)

        run_analysis(current_data_path, current_save_dir)

        # 【最高级别清理】：跑完一个 CSV 后，强制清理所有碎片
        gc.collect()