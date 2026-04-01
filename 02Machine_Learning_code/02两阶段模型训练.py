import pandas as pd
import numpy as np
import joblib
import os
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from datetime import datetime

# ================= 1. 基础配置 =================
DATA_PATH = r"E:\01Output\Experiments_new\Train_Data_Exp2_Time.csv"
SAVE_DIR = r"E:\01Output\Experiments_new\Model_Output\Final_Paper_Result"
os.makedirs(SAVE_DIR, exist_ok=True)

# 特征定义
features_base = ["Longitude", "Latitude", "DOY", "hour", "DEM_5km", "Pop_5km", "Roads_5km",
                 "CLCD_1_Cropland_Fraction_5km", "CLCD_2_Forest_Fraction_5km", "CLCD_5_Water_Fraction_5km",
                 "CLCD_8_Impervious_Fraction_5km", "blh", "rh", "t2m", "sp", "u10", "v10", "NDVI"]


# ================= 2. 工具函数与类 =================
class tqdm_callback:
    def __init__(self, total, desc):
        self.pbar = tqdm(total=total, desc=desc, unit="round")

    def __call__(self, env):
        self.pbar.update(1)
        if (env.iteration + 1) % 50 == 0:
            self.pbar.set_postfix({"RMSE": f"{env.evaluation_result_list[0][2]:.4f}"})


def plot_paper_scatter(y_true, y_pred, title, filename):
    print(f"🎨 正在生成学术散点图: {title}...")
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 抽样绘图
    if len(y_true) > 100000:
        idx = np.random.choice(len(y_true), 100000, replace=False)
        y_true_s, y_pred_s = y_true[idx], y_pred[idx]
    else:
        y_true_s, y_pred_s = y_true, y_pred

    plt.figure(figsize=(8, 7), dpi=150)
    hb = plt.hexbin(y_true_s, y_pred_s, gridsize=60, cmap='Spectral_r', mincnt=1)
    plt.colorbar(hb, label='Point Density')

    # 动态调整坐标轴范围（排除极个别未清洗干净的离群点干扰）
    lim_max = np.percentile(y_true_s, 99.9) * 1.1
    plt.xlim(-10, lim_max);
    plt.ylim(-10, lim_max)
    plt.plot([0, lim_max], [0, lim_max], 'r--', lw=2, label="1:1 Line")

    stats = f"$R^2 = {r2:.3f}$\n$MAE = {mae:.2f}$\n$RMSE = {rmse:.2f}$\n$N = {len(y_true):,}$"
    plt.gca().text(0.05, 0.95, stats, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel("Observed PM2.5 ($\mu g/m^3$)", fontsize=12)
    plt.ylabel("Predicted PM2.5 ($\mu g/m^3$)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, filename), bbox_inches='tight')
    plt.show()
    plt.close()


# ================= 3. 主程序 =================
def main():
    start_t = datetime.now()

    # --- 3.1 数据加载与物理过滤 ---
    print("📂 载入原始数据...")
    df = pd.read_csv(DATA_PATH)
    target_col = 'PM25_5030' if 'PM25_5030' in df.columns else 'PM25'
    aod_col = next((c for c in ['AOD', 'AOD_550', 'AOD_3km'] if c in df.columns), 'AOD')

    print("🧹 启动物理阈值清洗 (踢除离群点)...")
    initial_len = len(df)
    # 1. 剔除 PM2.5 异常值 (0-1000范围内视为有效)
    df = df[(df[target_col] >= 0) & (df[target_col] <= 1000)]
    # 2. 剔除 AOD 异常值 (AOD一般不大于10，999等属于填充码)
    df = df[((df[aod_col] >= 0) & (df[aod_col] <= 10)) | (df[aod_col].isna())]
    print(f"✅ 清洗完毕：剔除异常记录 {initial_len - len(df):,} 条，剩余有效样本 {len(df):,} 条。")

    # --- 3.2 特征处理 ---
    if 'DOY' not in df.columns:
        print("⏰ 提取时间特征...")
        t_col = next((c for c in ['UTC_Time', 'time', 'DateTime'] if c in df.columns), None)
        df[t_col] = pd.to_datetime(df[t_col])
        df['DOY'], df['hour'] = df[t_col].dt.dayofyear, df[t_col].dt.hour

    df['AOD_Flag'] = df[aod_col].notna().astype(int)
    df['AOD_Value'] = df[aod_col].fillna(0)
    features_res = features_base + ["AOD_Value", "AOD_Flag"]

    # --- 3.3 Stage 1: Base Model (背景建模) ---
    print(f"\n🧠 [Stage 1] 训练底色模型...")
    X1 = df[features_base];
    y1 = df[target_col]
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X1, y1, test_size=0.15, random_state=42)

    dtrain1 = lgb.Dataset(X_tr1, label=y_tr1)
    dtest1 = lgb.Dataset(X_te1, label=y_te1, reference=dtrain1)

    params1 = {'objective': 'regression', 'metric': 'rmse', 'device': 'gpu', 'learning_rate': 0.03, 'num_leaves': 511,
               'verbose': -1}
    cb1 = tqdm_callback(3000, "🔥 Stage 1 进化中")
    model_base = lgb.train(params1, dtrain1, num_boost_round=3000, valid_sets=[dtest1],
                           callbacks=[lgb.early_stopping(100), cb1])
    cb1.pbar.close()

    print("🧹 提取背景残差...")
    df['PM_Base_Pred'] = model_base.predict(X1)
    df['Residual_Label'] = df[target_col] - df['PM_Base_Pred']
    del dtrain1, dtest1;
    gc.collect()

    # --- 3.4 Stage 2: Residual Model (增益学习) ---
    print(f"\n🔥 [Stage 2] 训练残差修正模型...")
    X2 = df[features_res];
    y2 = df['Residual_Label']
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X2, y2, test_size=0.15, random_state=42)

    dtrain2 = lgb.Dataset(X_tr2, label=y_tr2)
    dtest2 = lgb.Dataset(X_te2, label=y_te2, reference=dtrain2)

    params2 = {'objective': 'regression', 'metric': 'rmse', 'device': 'gpu', 'learning_rate': 0.02, 'num_leaves': 1023,
               'verbose': -1, 'feature_fraction': 0.8}
    cb2 = tqdm_callback(5000, "⚡ Stage 2 进化中")
    model_res = lgb.train(params2, dtrain2, num_boost_round=5000, valid_sets=[dtest2],
                          callbacks=[lgb.early_stopping(100), cb2])
    cb2.pbar.close()

    # --- 3.5 最终融合评估 ---
    print("\n📊 正在进行最终融合评估...")
    res_pred = model_res.predict(X_te2)

    # 强制对齐：通过索引取回对应的原始值
    test_idx = X_te2.index
    y_true_final = df.loc[test_idx, target_col].values
    y_base_final = df.loc[test_idx, 'PM_Base_Pred'].values
    y_fusion_final = y_base_final + res_pred

    # 绘制最终融合图
    plot_paper_scatter(y_true_final, y_fusion_final, "Final Seamless Fusion PM2.5 (Cleaned)",
                       "Final_Fusion_Cleaned.png")

    # --- 3.6 保存 ---
    joblib.dump(model_base, os.path.join(SAVE_DIR, "Paper_Base_Model.pkl"))
    joblib.dump(model_res, os.path.join(SAVE_DIR, "Paper_Res_Model.pkl"))
    print(f"\n✅ 任务圆满完成！异常值已剔除，模型已优化。")
    print(f"⏱️ 总耗时: {datetime.now() - start_t}")


if __name__ == "__main__":
    main()