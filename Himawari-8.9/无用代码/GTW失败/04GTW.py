import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 核心 GTWR 库
from mgtwr.sel import SearchGTWRParameter
from mgtwr.model import GTWR

warnings.filterwarnings('ignore')

# 绘图配置：解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 变量配置 ---
target = "PM25_5030"
initial_features = [
    "AOD", "t2m", "blh", "rh", "sp", "u10", "v10", "NDVI",
    "DEM_5km", "Pop_5km", "Roads_5km",
    "CLCD_1_Cropland_Fraction_5km", "CLCD_2_Forest_Fraction_5km",
    "CLCD_5_Water_Fraction_5km", "CLCD_8_Impervious_Fraction_5km"
]
# 保护名单：核心物理/人类活动特征，不予剔除
protected_features = ["AOD", "CLCD_8_Impervious_Fraction_5km"]


def auto_vif_filter(df_sample, features_list, threshold=10.0):
    """逐步 VIF 筛选"""
    current_features = features_list.copy()
    removal_log = []
    while True:
        X_vif = df_sample[current_features].dropna()
        if X_vif.shape[1] <= 1: break
        vif_values = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        max_vif = max(vif_values)
        max_idx = vif_values.index(max_vif)
        worst_var = current_features[max_idx]

        if max_vif > threshold:
            if worst_var in protected_features:
                others_vif = [v if current_features[i] not in protected_features else 0 for i, v in
                              enumerate(vif_values)]
                if max(others_vif) > threshold:
                    idx_to_drop = others_vif.index(max(others_vif))
                    removal_log.append(f"【剔除】: {current_features[idx_to_drop]} (VIF={max(others_vif):.2f})")
                    current_features.pop(idx_to_drop)
                else:
                    break
            else:
                removal_log.append(f"【剔除】: {worst_var} (VIF={max_vif:.2f})")
                current_features.pop(max_idx)
        else:
            break
    return current_features, removal_log


def run_analysis(data_path, save_dir):
    file_name = os.path.basename(data_path)
    df = pd.read_csv(data_path)

    # 1. 预处理
    if 'u10' in df.columns and 'v10' in df.columns:
        df['WindSpeed'] = np.sqrt(df['u10'] ** 2 + df['v10'] ** 2)
    df['RealTime'] = pd.to_datetime(df['RealTime'])
    df['DOY'] = df['RealTime'].dt.dayofyear
    df['Month'] = df['RealTime'].dt.month

    actual_vars = [f for f in initial_features if f in df.columns]
    if 'WindSpeed' in df.columns: actual_vars.append('WindSpeed')

    # 提取有效样本
    df_train = df.dropna(subset=[target, "AOD"] + actual_vars).copy()
    if len(df_train) < 100: return

    # 抽样 (针对 GTWR 计算压力)
    df_sample = df_train.sample(n=min(len(df_train), 2000), random_state=42)
    final_features, drop_log = auto_vif_filter(df_sample, actual_vars)

    # 保存日志
    with open(os.path.join(save_dir, "变量筛选逻辑.txt"), "w", encoding="utf-8-sig") as f:
        f.write(f"实验集: {file_name}\n特征: {final_features}\n" + "\n".join(drop_log))

    # 2. GTWR 核心建模
    coords = df_sample[['Longitude', 'Latitude']].values
    t = df_sample['DOY'].values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sample[final_features].values)
    y = df_sample[target].values.reshape(-1, 1)

    try:
        # 带宽搜索
        sel = SearchGTWRParameter(coords, t, X_scaled, y, kernel='bisquare', fixed=False)
        bw, tau = sel.search(verbose=True)
        model = GTWR(coords, t, X_scaled, y, bw, tau, kernel='bisquare', fixed=False).fit()

        # 结果整理 (修复 predy 报错)
        y_pred = model.y.flatten()
        beta_names = ['Intercept'] + [f'Beta_{f}' for f in final_features]
        df_res = pd.concat([df_sample.reset_index(drop=True), pd.DataFrame(model.betas, columns=beta_names)], axis=1)
        df_res['Predict_PM25'] = y_pred
        df_res.to_csv(os.path.join(save_dir, "result_full.csv"), index=False, encoding='utf-8-sig')

        # --- 绘图模块 ---

        # 01. 验证散点图
        plt.figure(figsize=(7, 6))
        sns.regplot(x=y.flatten(), y=y_pred, scatter_kws={'alpha': 0.3, 's': 15}, line_kws={'color': 'red'})
        plt.title(f'验证散点图: {file_name}\nR² = {model.R2:.3f}')
        plt.savefig(os.path.join(save_dir, "01_验证散点图.png"), dpi=300)
        plt.close()

        # 02. 【新增】变量贡献度占比柱状图
        # 计算逻辑：各变量标准化系数的平均绝对值占比
        beta_means = np.abs(model.betas[:, 1:]).mean(axis=0)  # 排除截距
        contributions = (beta_means / beta_means.sum()) * 100

        plt.figure(figsize=(10, 6))
        # 排序后绘图
        contrib_df = pd.DataFrame({'Feature': final_features, 'Contrib': contributions})
        contrib_df = contrib_df.sort_values(by='Contrib', ascending=False)

        sns.barplot(x='Contrib', y='Feature', data=contrib_df, palette='viridis')
        for i, v in enumerate(contrib_df['Contrib']):
            plt.text(v + 0.5, i, f'{v:.1f}%', va='center')

        plt.title(f'各解释变量对 PM2.5 的平均贡献占比 ({file_name})')
        plt.xlabel('贡献度占比 (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "02_变量贡献占比图.png"), dpi=300)
        plt.close()

        # 03. AOD 贡献系数时间轨迹图 (针对 7 个站点)
        if 'Beta_AOD' in df_res.columns:
            plt.figure(figsize=(12, 6))
            stations = df_res['District'].unique()
            for station in stations:
                st_data = df_res[df_res['District'] == station].sort_values('DOY')
                plt.plot(st_data['DOY'], st_data['Beta_AOD'], label=station, alpha=0.8)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="站点")
            plt.title('AOD 贡献系数随时间演化图')
            plt.xlabel('一年中的天数 (DOY)')
            plt.ylabel('Beta_AOD')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "03_AOD时间轨迹图.png"), dpi=300)
            plt.close()

        # 04. AOD 月度敏感度箱线图
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Month', y='Beta_AOD', data=df_res, palette='coolwarm')
        plt.title('AOD 敏感度月度分布特征 (反映季节异质性)')
        plt.savefig(os.path.join(save_dir, "04_AOD月度箱线图.png"), dpi=300)
        plt.close()

    except Exception as e:
        print(f"\n❌ {file_name} 报错: {e}")


if __name__ == "__main__":
    exp_files = ["RF_Train_Data_Final.csv", "Train_Data_Exp2_Time.csv",
                 "Train_Data_Exp3_Space.csv", "Train_Data_Mix_PrioTime.csv", "Train_Data_Mix_PrioSpace.csv"]
    base_path = r"/Data/Output/Experiments"

    for file_name in tqdm(exp_files, desc="整体实验进度"):
        current_data_path = os.path.join(base_path, file_name)
        current_save_dir = os.path.join(base_path, "Model_Output", "04GTWR_Time", file_name.replace(".csv", ""))
        os.makedirs(current_save_dir, exist_ok=True)
        run_analysis(current_data_path, current_save_dir)