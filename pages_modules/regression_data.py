import numpy as np


ALGORITHM_DATASET_OPTIONS = {
    "linear": ["linear_trend", "linear_outliers"],
    "poly": ["poly_quadratic", "poly_cubic", "poly_sine"],
    "ridge": ["ridge_correlated", "ridge_dense_noise"],
    "lasso": ["lasso_sparse_signal", "lasso_correlated_noise"],
    "svr": ["svr_wave_band", "svr_noisy_curve"],
    "tree": ["tree_piecewise", "tree_local_steps"],
    "rf": ["rf_piecewise_noise", "rf_wave_ensemble"],
}


DATASET_META = {
    "linear_trend": {
        "label": "线性趋势噪声数据",
        "summary": "整体趋势接近一条直线，适合观察如何在噪声中找到平均变化方向。",
    },
    "linear_outliers": {
        "label": "含异常点的线性数据",
        "summary": "主趋势仍然线性，但少量异常点会明显拉动拟合直线，便于讲解线性回归对异常值较敏感。",
    },
    "poly_quadratic": {
        "label": "二次曲线拟合数据",
        "summary": "真实关系是平滑的抛物线，适合观察多项式阶数过低或过高时的拟合差异。",
    },
    "poly_cubic": {
        "label": "三次弯折曲线数据",
        "summary": "曲线同时包含上升和下降段，能更直观看到多项式回归对复杂趋势的表达能力。",
    },
    "poly_sine": {
        "label": "正弦波动数据",
        "summary": "数据带有周期波动，适合展示高阶多项式虽然能追曲线，但不一定最稳健。",
    },
    "ridge_correlated": {
        "label": "相关特征正则化数据",
        "summary": "多项特征彼此相关，适合观察 Ridge 如何通过系数收缩让模型更稳。",
    },
    "ridge_dense_noise": {
        "label": "高噪声相关特征数据",
        "summary": "相关特征更多、噪声更大，更容易体现正则化对过拟合的抑制作用。",
    },
    "lasso_sparse_signal": {
        "label": "稀疏有效特征数据",
        "summary": "只有少数特征真正有用，适合观察 Lasso 如何把不重要的系数压到接近 0。",
    },
    "lasso_correlated_noise": {
        "label": "相关噪声特征数据",
        "summary": "有效特征混在相关噪声里，适合比较 Lasso 的特征筛选效果。",
    },
    "svr_wave_band": {
        "label": "平滑波动回归数据",
        "summary": "数据整体非线性但变化平滑，适合观察 SVR 的 epsilon 容忍带与核函数效果。",
    },
    "svr_noisy_curve": {
        "label": "带噪声的非线性曲线",
        "summary": "曲线局部起伏明显，适合观察 SVR 在噪声下如何保持相对平滑的拟合。",
    },
    "tree_piecewise": {
        "label": "分段函数回归数据",
        "summary": "真实关系具有明显分段结构，特别适合展示决策树回归的阶梯状预测。",
    },
    "tree_local_steps": {
        "label": "局部突变回归数据",
        "summary": "目标值在局部区域突然变化，能直观看出树深度变化对分段逼近的影响。",
    },
    "rf_piecewise_noise": {
        "label": "带噪声的分段波动数据",
        "summary": "单棵树容易跟着局部噪声摆动，随机森林能通过平均投票得到更稳的拟合曲线。",
    },
    "rf_wave_ensemble": {
        "label": "复杂波动集成数据",
        "summary": "曲线既有平滑趋势又有局部起伏，适合比较单树与森林在稳定性上的差异。",
    },
}


DEFAULT_DATASET_BY_ALGORITHM = {
    "linear": "linear_trend",
    "poly": "poly_quadratic",
    "ridge": "ridge_correlated",
    "lasso": "lasso_sparse_signal",
    "svr": "svr_wave_band",
    "tree": "tree_piecewise",
    "rf": "rf_piecewise_noise",
}


FEATURE_NAMES_REGULARIZED = ["x", "相关线性", "x²", "相关 x²", "sin(x)", "相关 sin(x)", "cos(x)", "x³"]


def get_dataset_options(algorithm_key):
    return ALGORITHM_DATASET_OPTIONS[algorithm_key]


def get_default_dataset(algorithm_key):
    return DEFAULT_DATASET_BY_ALGORITHM[algorithm_key]


def get_dataset_label(dataset_key):
    return DATASET_META[dataset_key]["label"]


def get_dataset_summary(dataset_key):
    return DATASET_META[dataset_key]["summary"]


def generate_regression_dataset(dataset_key, n_samples, noise, random_state):
    rng = np.random.RandomState(random_state)
    x = np.sort(rng.uniform(-3.2, 3.2, size=n_samples))

    if dataset_key == "linear_trend":
        signal = 1.45 * x + 0.85
        y = signal + rng.normal(scale=0.45 + noise * 1.2, size=n_samples)
        X = x.reshape(-1, 1)
    elif dataset_key == "linear_outliers":
        signal = 1.30 * x + 1.10
        y = signal + rng.normal(scale=0.42 + noise * 1.15, size=n_samples)
        outlier_count = max(4, n_samples // 18)
        outlier_index = rng.choice(n_samples, outlier_count, replace=False)
        y[outlier_index] += rng.choice([-1.0, 1.0], size=outlier_count) * (1.8 + noise * 5.0)
        X = x.reshape(-1, 1)
    elif dataset_key == "poly_quadratic":
        signal = 0.42 * x ** 2 - 1.15 * x + 1.10
        y = signal + rng.normal(scale=0.40 + noise * 1.3, size=n_samples)
        X = x.reshape(-1, 1)
    elif dataset_key == "poly_cubic":
        signal = 0.16 * x ** 3 - 0.70 * x ** 2 - 0.35 * x + 1.0
        y = signal + rng.normal(scale=0.44 + noise * 1.2, size=n_samples)
        X = x.reshape(-1, 1)
    elif dataset_key == "poly_sine":
        signal = 1.55 * np.sin(1.18 * x) + 0.16 * x ** 2 - 0.35 * x
        y = signal + rng.normal(scale=0.34 + noise * 1.15, size=n_samples)
        X = x.reshape(-1, 1)
    elif dataset_key in ["ridge_correlated", "ridge_dense_noise", "lasso_sparse_signal", "lasso_correlated_noise"]:
        X = build_feature_matrix(dataset_key, x)
        weights = regularized_weights(dataset_key)
        signal = X.dot(weights)
        noise_scale = 0.36 + noise * (1.6 if dataset_key in ["ridge_dense_noise", "lasso_correlated_noise"] else 1.0)
        y = signal + rng.normal(scale=noise_scale, size=n_samples)
    elif dataset_key == "svr_wave_band":
        signal = 1.20 * np.sin(1.35 * x) + 0.35 * np.cos(2.7 * x) + 0.08 * x
        y = signal + rng.normal(scale=0.28 + noise * 0.95, size=n_samples)
        X = x.reshape(-1, 1)
    elif dataset_key == "svr_noisy_curve":
        signal = 0.85 * np.sin(1.8 * x) + 0.18 * x ** 2 - 0.55
        y = signal + rng.normal(scale=0.38 + noise * 1.25, size=n_samples)
        X = x.reshape(-1, 1)
    elif dataset_key == "tree_piecewise":
        signal = piecewise_signal(x)
        y = signal + rng.normal(scale=0.22 + noise * 0.85, size=n_samples)
        X = x.reshape(-1, 1)
    elif dataset_key == "tree_local_steps":
        signal = local_step_signal(x)
        y = signal + rng.normal(scale=0.20 + noise * 0.95, size=n_samples)
        X = x.reshape(-1, 1)
    elif dataset_key == "rf_piecewise_noise":
        signal = piecewise_signal(x) + 0.35 * np.sin(2.1 * x)
        y = signal + rng.normal(scale=0.34 + noise * 1.2, size=n_samples)
        X = x.reshape(-1, 1)
    else:
        signal = 1.05 * np.sin(1.15 * x) + 0.23 * x ** 2 - 0.52 * np.cos(2.6 * x)
        y = signal + rng.normal(scale=0.32 + noise * 1.15, size=n_samples)
        X = x.reshape(-1, 1)

    return X, y, {"plot_x": x, "true_signal": signal, "feature_names": feature_names_for_dataset(dataset_key)}


def build_feature_matrix(dataset_key, x_values):
    x = np.asarray(x_values, dtype=float)
    return np.column_stack(
        [
            x,
            0.90 * x + 0.18 * np.sin(1.8 * x),
            (x ** 2) / 3.2,
            (x ** 2) / 3.5 + 0.12 * np.cos(1.5 * x),
            np.sin(x),
            np.sin(x + 0.35),
            np.cos(x),
            (x ** 3) / 12.0,
        ]
    )


def regularized_weights(dataset_key):
    if dataset_key == "ridge_correlated":
        return np.array([1.20, 1.05, -0.95, -0.65, 0.85, 0.50, -0.35, 0.18], dtype=float)
    if dataset_key == "ridge_dense_noise":
        return np.array([1.15, 1.08, -0.92, -0.58, 0.72, 0.42, -0.28, 0.24], dtype=float)
    if dataset_key == "lasso_sparse_signal":
        return np.array([1.45, 0.0, -1.20, 0.0, 0.95, 0.0, 0.0, 0.0], dtype=float)
    return np.array([1.30, 0.0, -0.95, 0.0, 0.82, 0.0, -0.12, 0.0], dtype=float)


def feature_names_for_dataset(dataset_key):
    if dataset_key in ["ridge_correlated", "ridge_dense_noise", "lasso_sparse_signal", "lasso_correlated_noise"]:
        return FEATURE_NAMES_REGULARIZED
    return ["x"]


def reference_signal(dataset_key, x_values):
    x = np.asarray(x_values, dtype=float)
    if dataset_key == "linear_trend":
        return 1.45 * x + 0.85
    if dataset_key == "linear_outliers":
        return 1.30 * x + 1.10
    if dataset_key == "poly_quadratic":
        return 0.42 * x ** 2 - 1.15 * x + 1.10
    if dataset_key == "poly_cubic":
        return 0.16 * x ** 3 - 0.70 * x ** 2 - 0.35 * x + 1.0
    if dataset_key == "poly_sine":
        return 1.55 * np.sin(1.18 * x) + 0.16 * x ** 2 - 0.35 * x
    if dataset_key in ["ridge_correlated", "ridge_dense_noise", "lasso_sparse_signal", "lasso_correlated_noise"]:
        return build_feature_matrix(dataset_key, x).dot(regularized_weights(dataset_key))
    if dataset_key == "svr_wave_band":
        return 1.20 * np.sin(1.35 * x) + 0.35 * np.cos(2.7 * x) + 0.08 * x
    if dataset_key == "svr_noisy_curve":
        return 0.85 * np.sin(1.8 * x) + 0.18 * x ** 2 - 0.55
    if dataset_key == "tree_piecewise":
        return piecewise_signal(x)
    if dataset_key == "tree_local_steps":
        return local_step_signal(x)
    if dataset_key == "rf_piecewise_noise":
        return piecewise_signal(x) + 0.35 * np.sin(2.1 * x)
    return 1.05 * np.sin(1.15 * x) + 0.23 * x ** 2 - 0.52 * np.cos(2.6 * x)


def build_plot_features(dataset_key, x_values):
    if dataset_key in ["ridge_correlated", "ridge_dense_noise", "lasso_sparse_signal", "lasso_correlated_noise"]:
        return build_feature_matrix(dataset_key, x_values)
    return np.asarray(x_values, dtype=float).reshape(-1, 1)


def piecewise_signal(x):
    signal = np.empty_like(x)
    signal[x < -1.7] = -2.0 + 0.20 * x[x < -1.7]
    middle_left = (x >= -1.7) & (x < -0.4)
    signal[middle_left] = -0.8 + 0.55 * x[middle_left]
    middle = (x >= -0.4) & (x < 1.1)
    signal[middle] = 0.95 - 0.22 * x[middle]
    right = x >= 1.1
    signal[right] = 1.35 + 0.38 * np.sin(1.2 * x[right])
    return signal


def local_step_signal(x):
    signal = np.empty_like(x)
    signal[x < -1.5] = -1.8
    region_1 = (x >= -1.5) & (x < -0.2)
    signal[region_1] = -0.1 + 0.35 * np.sin(3.0 * x[region_1])
    region_2 = (x >= -0.2) & (x < 1.2)
    signal[region_2] = 1.15
    signal[x >= 1.2] = 0.35
    return signal


def split_indices(sample_count, test_size, random_state):
    rng = np.random.RandomState(random_state)
    indices = np.arange(sample_count)
    rng.shuffle(indices)
    test_count = max(1, int(sample_count * test_size))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return train_indices, test_indices
