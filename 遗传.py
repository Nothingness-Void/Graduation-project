import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.model_selection as _ms
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import sys
import joblib
import sklearn.utils
from multiprocessing import freeze_support

# Compatibility shim for genetic_selection on newer scikit-learn
if not hasattr(sklearn.utils, "_joblib"):
    sklearn.utils._joblib = joblib
    sys.modules["sklearn.utils._joblib"] = joblib

# Compatibility shim: genetic_selection passes fit_params to cross_val_score
if not hasattr(_ms.cross_val_score, "_ga_compat"):
    _orig_cvs = _ms.cross_val_score

    def _cross_val_score_compat(*args, **kwargs):
        kwargs.pop("fit_params", None)
        return _orig_cvs(*args, **kwargs)

    _cross_val_score_compat._ga_compat = True
    _ms.cross_val_score = _cross_val_score_compat

from genetic_selection import GeneticSelectionCV

def main():
    # 1. 读取经过清洗的数据 (使用你之前筛选出的16个特征)
    # 建议直接读取 features_optimized.xlsx，或者手动指定那16个列
    feature_cols = [
        'MolWt1', 'logP1', 'TPSA1', 'MaxAbsPartialCharge1', 'LabuteASA1',
        'logP2', 'MaxAbsPartialCharge2', 'LabuteASA2',
        'Avalon Similarity', 'Morgan Similarity',
        'Delta_LogP', 'Delta_TPSA', 'HB_Match', 'Delta_MolMR',
        'CSP3_2', 'Inv_T'
    ]

    data = pd.read_excel('data/features_optimized.xlsx')
    X = data[feature_cols]
    y = data['χ-result']

    # 划分数据 (保持和之前一样的随机种子，方便对比)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化 (GA 对数值敏感，必须标准化)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ============================================================
    # 核心部分：配置遗传算法
    # ============================================================
    # 这里的 estimator 可以换。
    # 策略 A: LinearRegression -> 寻找最简单的物理公式 (GA-MLR) -> 解释性最强
    # 策略 B: RandomForest -> 寻找最强的非线性组合 -> 精度最高
    estimator = LinearRegression()
    # estimator = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    clf = RandomForestRegressor(
    n_estimators=50,   # 进化阶段 50 棵树足够了，节省计算时间
    max_depth=5,       # 限制深度防止特征选择阶段就过拟合
    n_jobs=-1,
    random_state=42
)
    print(f"正在启动遗传算法进化，使用模型: {estimator.__class__.__name__} ...")
    print("这可能需要几分钟，请耐心等待生物进化...")

    selector = GeneticSelectionCV(
        clf,
        cv=5,
        verbose=1,
        scoring="r2",
        max_features=10,  # 限制最多选10个特征 (防止过拟合)
        n_population=200, # 种群大小：一次养200个模型
        crossover_proba=0.5, # 杂交率
        mutation_proba=0.2,  # 变异率 (重要！防止近亲繁殖)
        n_generations=50, # 进化代数：繁衍50代
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.05,
        tournament_size=3,
        n_gen_no_change=10, # 如果10代没有进化，提前结束
        caching=True,
        n_jobs=-1
    )

    # 开始进化
    selector = selector.fit(X_train_scaled, y_train)

    # ============================================================
    # 结果分析
    # ============================================================

    # 获取被选中的特征
    selected_features = X.columns[selector.support_]
    print("\n" + "="*50)
    print("🎉 进化完成！自然选择的结果：")
    print("="*50)
    print(f"保留了 {len(selected_features)} 个特征：")
    print(list(selected_features))

    # 在测试集上验证
    # 注意：必须只用选出来的特征去预测
    X_train_sel = selector.transform(X_train_scaled)
    X_test_sel = selector.transform(X_test_scaled)

    # 重新训练最终模型
    estimator.fit(X_train_sel, y_train)
    y_pred = estimator.predict(X_test_sel)

    final_r2 = r2_score(y_test, y_pred)
    print(f"\n最终模型 Test R2: {final_r2:.4f}")

    # 如果你用的是线性回归，还可以打印出公式
    if isinstance(estimator, LinearRegression):
        print("\n推导出的物理公式：")
        formula = "χ ≈ {:.4f}".format(estimator.intercept_)
        for weight, feat in zip(estimator.coef_, selected_features):
            sign = "+" if weight >= 0 else "-"
            formula += f" {sign} {abs(weight):.4f}*{feat}"
        print(formula)


if __name__ == '__main__':
    freeze_support()
    main()