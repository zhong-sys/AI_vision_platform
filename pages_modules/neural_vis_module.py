"""
NeuralVis 神经网络可视化训练模块
================================
将 NeuralVis 桌面应用的核心算法与交互逻辑迁移至 Streamlit 框架。

包含：
- SimpleNN      : 全连接神经网络核心类
- generate_dataset : 数据集生成函数
- Trainer       : 训练控制器
- render_neural_network_viz : Streamlit 主渲染入口
"""

import os
import time

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 直接指定云端和本地都存在的字体路径
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if not os.path.exists(font_path):
    # 本地 Windows 的回退
    font_path = 'C:/Windows/Fonts/msyh.ttc'

if os.path.exists(font_path):
    from matplotlib.font_manager import FontProperties
    fp = FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fp.get_name()
else:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
# =============================================================================
# 1. 核心算法类（从 NeuralVis 移植，保留完整功能，禁止修改算法逻辑）
# =============================================================================

class SimpleNN:
    """简单的全连接神经网络，用于教学演示。支持切换激活函数与损失函数。"""

    def __init__(self, layer_sizes: list, activation: str = "relu", loss_type: str = "cross_entropy"):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.activation_name = activation
        self.loss_type = loss_type

        self.W = []
        self.b = []
        for i in range(self.n_layers - 1):
            if activation == "relu":
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:
                scale = np.sqrt(1.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.W.append(w)
            self.b.append(b)

        self.z = []
        self.a = []
        self.grads_W = [np.zeros_like(w) for w in self.W]
        self.grads_b = [np.zeros_like(b) for b in self.b]

    def _activation(self, x):
        if self.activation_name == "relu":
            return np.maximum(0, x)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_name == "tanh":
            return np.tanh(x)
        elif self.activation_name == "linear":
            return x
        else:
            return x

    def _activation_derivative(self, x):
        if self.activation_name == "relu":
            return (x > 0).astype(float)
        elif self.activation_name == "sigmoid":
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif self.activation_name == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation_name == "linear":
            return np.ones_like(x)
        else:
            return np.ones_like(x)

    def _output_activation(self, x):
        if self.loss_type == "cross_entropy":
            if self.layer_sizes[-1] > 2:
                # 多分类：softmax
                exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return exp_x / np.sum(exp_x, axis=1, keepdims=True)
            else:
                return 1 / (1 + np.exp(-x))
        else:
            return x

    def _output_activation_derivative(self, x):
        if self.loss_type == "cross_entropy":
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        else:
            return np.ones_like(x)

    def forward(self, X):
        self.a = [X]
        self.z = []
        out = X
        for i in range(self.n_layers - 1):
            z = out @ self.W[i] + self.b[i]
            self.z.append(z)
            if i < self.n_layers - 2:
                out = self._activation(z)
            else:
                out = self._output_activation(z)
            self.a.append(out)
        return out

    def backward(self, X, y):
        m = X.shape[0]
        a_out = self.a[-1]
        z_out = self.z[-1]

        if self.loss_type == "cross_entropy":
            delta = a_out - y
        elif self.loss_type == "mse":
            delta = (a_out - y) * 2 * self._output_activation_derivative(z_out)
        elif self.loss_type == "mae":
            delta = np.sign(a_out - y) * self._output_activation_derivative(z_out)
        else:
            delta = a_out - y

        grads_W = []
        grads_b = []
        for i in range(self.n_layers - 2, -1, -1):
            dW = (self.a[i].T @ delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            grads_W.insert(0, dW)
            grads_b.insert(0, db)

            if i > 0:
                delta = (delta @ self.W[i].T) * self._activation_derivative(self.z[i - 1])

        self.grads_W = grads_W
        self.grads_b = grads_b
        return grads_W, grads_b

    def update(self, lr: float):
        for i in range(self.n_layers - 1):
            self.W[i] -= lr * self.grads_W[i]
            self.b[i] -= lr * self.grads_b[i]

    def get_weights(self):
        return self.W

    def get_biases(self):
        return self.b

    def get_gradients(self):
        return self.grads_W, self.grads_b


def generate_dataset(task: str, input_dim: int = 2, n_samples: int = 200, noise: float = 0.1, output_size: int = None):
    """生成示例数据集，特征维度自动匹配输入层神经元数。"""
    rng = np.random.default_rng(42)

    if task == "分类":
        n = n_samples // 2
        if input_dim == 2:
            r1 = rng.normal(2.0, 0.5, n)
            theta1 = rng.uniform(0, 2 * np.pi, n)
            x1 = np.column_stack((r1 * np.cos(theta1), r1 * np.sin(theta1)))
            r2 = rng.normal(5.0, 0.8, n)
            theta2 = rng.uniform(0, 2 * np.pi, n)
            x2 = np.column_stack((r2 * np.cos(theta2), r2 * np.sin(theta2)))
        else:
            c1 = rng.normal(0, 1, input_dim)
            c2 = rng.normal(2, 1, input_dim)
            x1 = rng.multivariate_normal(c1, np.eye(input_dim) * 0.5, n)
            x2 = rng.multivariate_normal(c2, np.eye(input_dim) * 0.8, n)

        y1 = np.zeros(n)
        y2 = np.ones(n)
        X = np.vstack((x1, x2))
        y = np.hstack((y1, y2)).reshape(-1, 1)
        return X, y

    elif task == "回归":
        X = rng.uniform(-3, 3, (n_samples, input_dim))
        y = np.sin(X[:, 0]) + rng.normal(0, noise, n_samples)
        return X, y.reshape(-1, 1)

    elif task == "聚类":
        n_per_class = n_samples // 3
        if input_dim == 2:
            centers = np.array([[-2, -2], [2, 2], [2, -2]])
        else:
            centers = rng.uniform(-3, 3, (3, input_dim))
        X = []
        y = []
        for i, c in enumerate(centers):
            pts = rng.multivariate_normal(c, np.eye(input_dim) * 0.5, n_per_class)
            X.append(pts)
            y.append(np.full(len(pts), i))
        X = np.vstack(X)
        y = np.hstack(y).reshape(-1, 1)
        return X, y

    else:
        raise ValueError(f"Unknown task: {task}")


class Trainer:
    """训练控制器，支持激活函数与损失函数切换。"""

    def __init__(
        self,
        task: str = "分类",
        layer_sizes: list = None,
        activation: str = "relu",
        loss_type: str = "cross_entropy",
        lr: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        self.task = task
        self.layer_sizes = layer_sizes or [2, 4, 1]
        self.activation = activation
        self.loss_type = loss_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.nn = SimpleNN(self.layer_sizes, activation=activation, loss_type=loss_type)
        self.X, self.y = generate_dataset(task, input_dim=self.layer_sizes[0], output_size=self.layer_sizes[-1])
        self.current_epoch = 0
        self.loss_history = []
        self.acc_history = []

    def reset(self):
        self.nn = SimpleNN(self.layer_sizes, activation=self.activation, loss_type=self.loss_type)
        self.X, self.y = generate_dataset(self.task, input_dim=self.layer_sizes[0], output_size=self.layer_sizes[-1])
        self.current_epoch = 0
        self.loss_history.clear()
        self.acc_history.clear()

    def step(self):
        """单步训练，返回包含可视化所需全部信息的字典。"""
        output = self.nn.forward(self.X)
        loss = self._compute_loss(output, self.y)
        self.nn.backward(self.X, self.y)
        self.nn.update(self.lr)

        self.loss_history.append(float(loss))
        self.current_epoch += 1

        acc = None
        if self.task == "分类":
            if output.shape[1] > 1 and self.y.shape[1] > 1:
                acc = float(np.mean(np.argmax(output, axis=1) == np.argmax(self.y, axis=1)))
            else:
                acc = float(np.mean((output > 0.5).astype(int) == self.y))
            self.acc_history.append(acc)

        return {
            "epoch": self.current_epoch,
            "loss": float(loss),
            "accuracy": acc,
            "weights": [w.copy() for w in self.nn.W],
            "biases": [b.copy() for b in self.nn.b],
            "grads_W": [g.copy() for g in self.nn.grads_W],
            "grads_b": [g.copy() for g in self.nn.grads_b],
            "activations": [a.copy() for a in self.nn.a],
            "zs": [z.copy() for z in self.nn.z],
        }

    def _compute_loss(self, output, y):
        m = y.shape[0]
        eps = 1e-8
        if self.loss_type == "cross_entropy":
            if output.shape[1] > 1 and y.shape[1] > 1:
                # 多分类：y 是 one-hot，output 是 softmax 概率
                return -np.mean(np.sum(y * np.log(output + eps), axis=1))
            else:
                return -np.mean(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps))
        elif self.loss_type == "mse":
            return np.mean((output - y) ** 2)
        elif self.loss_type == "mae":
            return np.mean(np.abs(output - y))
        else:
            return -np.mean(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps))


# =============================================================================
# 2. 可视化辅助函数
# =============================================================================

def _get_fig_dpi():
    return 100


def plot_data_distribution(X, y):
    """绘制训练数据分布散点图。"""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=_get_fig_dpi())
    if X.shape[1] >= 2:
        scatter = ax.scatter(
            X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm",
            edgecolors="k", alpha=0.7, s=50
        )
        ax.set_xlabel("特征 1", fontsize=11, fontproperties=_cn_fp(11))
        ax.set_ylabel("特征 2", fontsize=11, fontproperties=_cn_fp(11))
        plt.colorbar(scatter, ax=ax, label="标签")
    else:
        ax.scatter(
            X[:, 0], y.ravel(), c=y.ravel(), cmap="coolwarm",
            edgecolors="k", alpha=0.7, s=50
        )
        ax.set_xlabel("特征 1", fontsize=11, fontproperties=_cn_fp(11))
        ax.set_ylabel("标签", fontsize=11, fontproperties=_cn_fp(11))
    ax.set_title("训练数据分布", color="#1A7EC1", fontproperties=_cn_fp(13, bold=True))
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def plot_training_curves(history):
    """绘制损失值与准确率随轮次变化的曲线。"""
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=_get_fig_dpi())
    epochs = history["epoch"]
    if not epochs:
        ax1.text(0.5, 0.5, "暂无训练数据", ha="center", va="center", color="#999", fontproperties=_cn_fp(14))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis("off")
        fig.patch.set_facecolor("white")
        return fig

    color_loss = "#1A7EC1"
    ax1.plot(epochs, history["loss"], color=color_loss, linewidth=2, label="损失值", marker="o", markersize=3)
    ax1.set_xlabel("轮次 (Epoch)", fontsize=11, fontproperties=_cn_fp(11))
    ax1.set_ylabel("损失值 (Loss)", color=color_loss, fontsize=11, fontproperties=_cn_fp(11))
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, alpha=0.3)

    if history["acc"]:
        ax2 = ax1.twinx()
        color_acc = "#F39C12"
        ax2.plot(epochs, history["acc"], color=color_acc, linewidth=2, label="准确率", marker="s", markersize=3)
        ax2.set_ylabel("准确率 (Accuracy)", color=color_acc, fontsize=11, fontproperties=_cn_fp(11))
        ax2.tick_params(axis="y", labelcolor=color_acc)
        ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", prop=_cn_fp(9))
    else:
        ax1.legend(loc="upper right", prop=_cn_fp(9))

    ax1.set_title("训练过程曲线", color="#1A7EC1", fontproperties=_cn_fp(13, bold=True))
    ax1.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def _get_neuron_color_forward(val, norm_val):
    """前向传播：蓝色系，亮度对应激活值大小。"""
    # 从 #1A7EC1 (26,126,193) 到更亮的蓝色
    r = 0.10 + norm_val * 0.15
    g = 0.35 + norm_val * 0.40
    b = 0.60 + norm_val * 0.35
    return (r, g, b)


def _get_neuron_color_backward(intensity):
    """反向传播：红色系，亮度对应梯度大小。"""
    # 从暗红到亮红
    r = 0.50 + intensity * 0.45
    g = 0.15 + intensity * 0.20
    b = 0.15 + intensity * 0.10
    return (r, g, b)


def _get_connection_color_weight(val, max_w):
    """权重可视化：正权重偏蓝，负权重偏红。"""
    intensity = abs(val) / max_w
    if val >= 0:
        r = 0.06 + intensity * 0.15
        g = 0.25 + intensity * 0.35
        b = 0.50 + intensity * 0.45
        return (r, g, b), 0.3 + intensity * 2.5, 0.25 + intensity * 0.60
    else:
        r = 0.55 + intensity * 0.40
        g = 0.15 + intensity * 0.15
        b = 0.15 + intensity * 0.10
        return (r, g, b), 0.3 + intensity * 2.5, 0.25 + intensity * 0.60


def _get_connection_color_forward(val, max_w):
    """前向传播连线：统一蓝色系，强度对应权重绝对值。"""
    intensity = abs(val) / max_w
    r = 0.10 + intensity * 0.10
    g = 0.35 + intensity * 0.35
    b = 0.60 + intensity * 0.35
    return (r, g, b), 0.3 + intensity * 2.5, 0.25 + intensity * 0.60


def plot_network_structure(layer_sizes, mode="weights", weights=None, activations=None, grads_W=None, figsize=None):
    """
    绘制神经网络拓扑结构。

    mode:
        "structure" - 仅显示拓扑，灰色默认
        "forward"   - 前向传播：神经元亮度对应激活值（蓝色系）
        "backward"  - 反向传播：神经元与连线对应梯度（红色系）
        "weights"   - 权重可视化：正权重偏蓝，负权重偏红，连线粗细映射权重绝对值
    """
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    if figsize is not None:
        fig_w, fig_h = figsize
    else:
        fig_w = max(8, n_layers * 2.2)
        fig_h = max(5, max_neurons * 0.32)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=_get_fig_dpi())

    x_positions = np.linspace(0.08, 0.92, n_layers)
    neuron_positions = []

    for i, size in enumerate(layer_sizes):
        x = x_positions[i]
        display_size = min(size, 20)
        if display_size == 1:
            y_positions = [0.5]
        else:
            y_positions = np.linspace(0.08, 0.92, display_size)
        neuron_positions.append([(x, y) for y in y_positions])

    # 绘制层间连线
    for i in range(n_layers - 1):
        if weights is not None and i < len(weights):
            w = weights[i]
            max_w = np.max(np.abs(w)) + 1e-8
        else:
            w = None
            max_w = 1.0

        prev_size = layer_sizes[i]
        curr_size = layer_sizes[i + 1]
        prev_display = min(prev_size, 20)
        curr_display = min(curr_size, 20)

        for j, (x1, y1) in enumerate(neuron_positions[i]):
            for k, (x2, y2) in enumerate(neuron_positions[i + 1]):
                if w is not None:
                    src_idx = j if prev_size <= 20 else int(j * prev_size / prev_display)
                    dst_idx = k if curr_size <= 20 else int(k * curr_size / curr_display)
                    src_idx = min(src_idx, w.shape[0] - 1)
                    dst_idx = min(dst_idx, w.shape[1] - 1)
                    val = w[src_idx, dst_idx]

                    if mode == "backward" and grads_W is not None and i < len(grads_W):
                        g = grads_W[i]
                        grad_val = g[src_idx, dst_idx]
                        max_g = np.max(np.abs(g)) + 1e-8
                        intensity = abs(grad_val) / max_g
                        color = _get_neuron_color_backward(intensity)
                        linewidth = 0.3 + intensity * 3.0
                        alpha = 0.30 + intensity * 0.55
                    elif mode == "weights":
                        color, linewidth, alpha = _get_connection_color_weight(val, max_w)
                    elif mode == "forward":
                        color, linewidth, alpha = _get_connection_color_forward(val, max_w)
                    else:
                        color = "#BBBBBB"
                        linewidth = 0.3
                        alpha = 0.3
                else:
                    color = "#BBBBBB"
                    linewidth = 0.3
                    alpha = 0.3
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha, zorder=1)

    # 绘制神经元节点
    for i, positions in enumerate(neuron_positions):
        size = layer_sizes[i]
        for j, (x, y) in enumerate(positions):
            facecolor = "#1A7EC1"
            edgecolor = "white"

            if mode == "forward" and activations is not None and i < len(activations):
                act = activations[i]
                if act.ndim > 1 and j < act.shape[1]:
                    val = float(act[0, j])
                    norm_val = min(abs(val), 1.0)
                    facecolor = _get_neuron_color_forward(val, norm_val)
                    edgecolor = "#94a3b8"

            elif mode == "backward" and grads_W is not None and i > 0:
                # 第 i 层的神经元梯度来自 grads_W[i-1] 的第 j 列（平均绝对值）
                g = grads_W[i - 1]
                if j < g.shape[1]:
                    per_neuron_grad = np.mean(np.abs(g[:, j]))
                    max_g = np.max(np.abs(g)) + 1e-8
                    intensity = float(per_neuron_grad) / max_g
                    facecolor = _get_neuron_color_backward(intensity)
                    edgecolor = "#fca5a5"

            elif mode == "weights" and weights is not None:
                # 权重视图下神经元显示默认蓝色，但如果有激活值则轻微着色
                if activations is not None and i < len(activations):
                    act = activations[i]
                    if act.ndim > 1 and j < act.shape[1]:
                        val = float(act[0, j])
                        norm_val = min(abs(val), 1.0)
                        facecolor = _get_neuron_color_forward(val, norm_val)

            circle = Circle(
                (x, y), 0.022,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.5,
                zorder=2,
            )
            ax.add_patch(circle)

            # 数值标签（前向传播时显示激活值，反向时显示梯度强度）
            if mode == "forward" and activations is not None and i < len(activations):
                act = activations[i]
                if act.ndim > 1 and j < act.shape[1]:
                    val = float(act[0, j])
                    ax.text(x + 0.03, y, f"{val:.2f}", fontsize=7, color="#333", va="center")

            elif mode == "backward" and grads_W is not None and i > 0:
                g = grads_W[i - 1]
                if j < g.shape[1]:
                    per_neuron_grad = np.mean(np.abs(g[:, j]))
                    max_g = np.max(np.abs(g)) + 1e-8
                    intensity = float(per_neuron_grad) / max_g
                    ax.text(x + 0.03, y, f"{intensity:.2f}", fontsize=7, color="#c0392b", va="center")

        # 层标签
        if i == 0:
            label = f"输入层\n({size} neurons)"
        elif i == n_layers - 1:
            label = f"输出层\n({size} neurons)"
        else:
            label = f"隐藏层 {i}\n({size} neurons)"
        ax.text(
            x, 0.01, label,
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="#1A7EC1", fontproperties=_cn_fp(10, bold=True)
        )

    # 模式标题
    mode_titles = {
        "structure": "网络拓扑结构",
        "forward": "前向传播可视化 — 神经元亮度对应激活值大小（蓝色系）",
        "backward": "反向传播可视化 — 神经元与连线对应梯度大小（红色系）",
        "weights": "权重可视化 — 正权重偏蓝，负权重偏红，连线粗细映射权重绝对值",
    }
    ax.set_title(mode_titles.get(mode, ""), fontsize=12, fontweight="bold", color="#0F5B9E", pad=10, fontproperties=_cn_fp(12, bold=True))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def plot_weight_heatmaps(weights):
    """绘制各层权重矩阵的热力图。"""
    n = len(weights)
    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "暂无权重数据", ha="center", va="center", color="#999", fontproperties=_cn_fp(14))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.patch.set_facecolor("white")
        return fig

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.8 * rows), dpi=_get_fig_dpi())
    if n == 1:
        axes = [axes]
    else:
        axes = np.atleast_1d(axes).flatten().tolist()

    for i, (ax, w) in enumerate(zip(axes, weights)):
        vmax = np.max(np.abs(w))
        im = ax.imshow(w, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"层 {i + 1} → {i + 2} 权重矩阵\n形状: {w.shape}", fontsize=10, color="#1A7EC1", fontproperties=_cn_fp(10))
        ax.set_xlabel("当前层神经元", fontsize=9, fontproperties=_cn_fp(9))
        ax.set_ylabel("前层神经元", fontsize=9, fontproperties=_cn_fp(9))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_facecolor("#F8FAFC")

    for ax in axes[n:]:
        ax.axis("off")

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


# =============================================================================
# 3. 原理讲解文本生成
# =============================================================================

def _get_theory_text(epoch: int) -> str:
    """根据当前训练轮次返回对应的原理讲解文本。"""
    if epoch == 0:
        return (
            "欢迎使用神经网络可视化学习工具！\n\n"
            "在右侧面板设置网络结构、激活函数、损失函数和学习率后，点击「构建网络」开始。"
        )
    elif epoch == 1:
        return (
            "【前向传播】数据从输入层经过隐藏层流向输出层，\n"
            "每层通过加权求和与激活函数进行非线性变换，最终得到预测值。"
        )
    elif epoch == 2:
        return (
            "【反向传播】计算预测值与真实值之间的误差（损失），\n"
            "然后从输出层向输入层逐层传递梯度，更新每个权重。"
        )
    elif epoch == 3:
        return (
            "【权重更新】根据梯度和学习率调整权重。\n"
            "学习率决定了每次更新的步长，太大可能震荡，太小收敛慢。"
        )
    else:
        return (
            "【整体循环】神经网络不断重复\n"
            "「前向传播 → 计算损失 → 反向传播 → 权重更新」的过程，直到损失收敛。"
        )


# =============================================================================
# 4. Session State 管理
# =============================================================================

def _init_session_state():
    """初始化 NeuralVis 模块所需的 session_state 变量。"""
    defaults = {
        "nv_trainer": None,
        "nv_history": {"loss": [], "acc": [], "epoch": []},
        "nv_logs": [],
        "nv_training": False,
        "nv_network_built": False,
        "nv_layer_sizes": None,
        "nv_target_epochs": 100,
        "nv_last_info": None,
        "nv_anim_phase": None,
        "nv_anim_info": None,
        "nv_viz_mode": "weights",
        "nv_auto_training": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =============================================================================
# 5. Streamlit 渲染函数（主入口）
# =============================================================================

def render_neural_network_viz():
    """渲染完整的神经网络可视化训练界面。主入口函数。"""
    _init_session_state()

    # 页面标题与说明
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span style="font-size: 22px; font-weight: bold; color: #0F5B9E;">🧬 基础神经网络可视化训练</span>
    </div>
    <div style="color: #555; font-size: 16px; margin-bottom: 16px;">
        通过调整网络结构、激活函数与损失函数，观察神经网络前向传播、反向传播与权重更新的完整过程。
    </div>
    """, unsafe_allow_html=True)

    # ---- 自动训练 ----
    if st.session_state.nv_auto_training and st.session_state.nv_trainer is not None:
        _execute_auto_training()
        st.session_state.nv_auto_training = False
        st.session_state.nv_viz_mode = "weights"
        st.rerun()

    left_col, right_col = st.columns([0.3, 0.7])

    # ==================== 左侧参数配置面板 ====================
    with left_col:
        st.markdown("<div class='sidebar-header'>⚙️ 参数配置</div>", unsafe_allow_html=True)

        # 网络结构参数
        input_size = st.slider("输入层神经元数", 2, 10, 2, key="nv_input_size")
        n_hidden = st.slider("隐藏层数量", 1, 4, 2, key="nv_n_hidden")

        hidden_layers = []
        for i in range(n_hidden):
            default_val = 8 if i < 2 else 4
            val = st.slider(f"第 {i + 1} 隐藏层神经元数", 2, 20, default_val, key=f"nv_hidden_{i}")
            hidden_layers.append(val)

        output_size = st.selectbox("输出层神经元数", [1, 2, 3, 4], index=0, key="nv_output_size")

        # 函数与训练参数
        activation_display = st.selectbox(
            "激活函数", ["ReLU", "Sigmoid", "Tanh", "Linear"],
            index=0, key="nv_activation_display"
        )
        loss_display = st.selectbox(
            "损失函数", ["交叉熵", "MSE", "MAE"],
            index=0, key="nv_loss_display"
        )
        lr = st.selectbox(
            "学习率", [0.001, 0.01, 0.1, 0.5],
            index=1, key="nv_lr", format_func=lambda x: f"{x}"
        )
        max_epochs = st.slider("最大训练轮数", 10, 1000, 100, key="nv_max_epochs")

        # 名称映射
        activation_map = {"ReLU": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh", "Linear": "linear"}
        loss_map = {"交叉熵": "cross_entropy", "MSE": "mse", "MAE": "mae"}
        activation = activation_map[activation_display]
        loss_type = loss_map[loss_display]

        # 操作按钮（4个，与原版 PySide6 对应）
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            build_clicked = st.button("🔧 构建网络", key="nv_btn_build", use_container_width=True)
        with btn_col2:
            reset_clicked = st.button("🔄 重置", key="nv_btn_reset", use_container_width=True)

        btn_col3, btn_col4 = st.columns(2)
        with btn_col3:
            step_clicked = st.button("👟 单步执行", key="nv_btn_step", use_container_width=True)
        with btn_col4:
            auto_clicked = st.button("▶️ 自动训练", key="nv_btn_auto", use_container_width=True)

        # ---- 处理按钮事件 ----

        # 构建网络
        if build_clicked:
            layer_sizes = [input_size] + hidden_layers + [output_size]
            st.session_state.nv_trainer = Trainer(
                task="分类",
                layer_sizes=layer_sizes,
                activation=activation,
                loss_type=loss_type,
                lr=lr,
                epochs=max_epochs,
                batch_size=32,
            )
            st.session_state.nv_history = {"loss": [], "acc": [], "epoch": []}
            st.session_state.nv_logs = []
            st.session_state.nv_network_built = True
            st.session_state.nv_layer_sizes = layer_sizes
            st.session_state.nv_target_epochs = max_epochs
            st.session_state.nv_last_info = None
            st.session_state.nv_anim_phase = None
            st.session_state.nv_anim_info = None
            st.session_state.nv_viz_mode = "weights"
            st.success(f"✅ 网络构建成功！结构: {layer_sizes}")
            st.info(_get_theory_text(0))

        # 重置
        if reset_clicked:
            st.session_state.nv_trainer = None
            st.session_state.nv_history = {"loss": [], "acc": [], "epoch": []}
            st.session_state.nv_logs = []
            st.session_state.nv_network_built = False
            st.session_state.nv_layer_sizes = None
            st.session_state.nv_target_epochs = 100
            st.session_state.nv_last_info = None
            st.session_state.nv_training = False
            st.session_state.nv_anim_phase = None
            st.session_state.nv_anim_info = None
            st.session_state.nv_viz_mode = "weights"
            st.info("🔄 状态已重置")

        # 单步执行
        if step_clicked:
            if not st.session_state.nv_network_built or st.session_state.nv_trainer is None:
                st.error("⚠️ 请先点击「构建网络」后再执行单步训练。")
            else:
                info = _do_single_step()
                if info is not None:
                    st.session_state.nv_anim_info = info
                    st.session_state.nv_anim_phase = 0
                    st.session_state.nv_viz_mode = "weights"
                    st.rerun()

        # 自动训练
        if auto_clicked:
            if not st.session_state.nv_network_built or st.session_state.nv_trainer is None:
                st.error("⚠️ 请先点击「构建网络」后再开始自动训练。")
            else:
                st.session_state.nv_target_epochs = max_epochs
                st.session_state.nv_anim_phase = None
                st.session_state.nv_anim_info = None
                st.session_state.nv_auto_training = True
                st.rerun()

    # ==================== 右侧可视化区域 ====================
    with right_col:
        tab_network, tab_curve = st.tabs([
            "🕸️ 网络结构", "📈 训练曲线"
        ])

        # ---------- 训练曲线 ----------
        with tab_curve:
            if st.session_state.nv_history["epoch"]:
                try:
                    fig = plot_training_curves(st.session_state.nv_history)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"绘制训练曲线时出错: {e}")
            else:
                st.info("开始训练后将在此显示损失值与准确率的变化曲线。")

        # ---------- 网络结构 ----------
        with tab_network:
            if st.session_state.nv_layer_sizes is not None:
                try:
                    anim_phase = st.session_state.nv_anim_phase
                    anim_info = st.session_state.nv_anim_info

                    if anim_phase is not None and anim_info is not None:
                        phase_config = {
                            0: {
                                "mode": "forward",
                                "title": "🔷 Step 1/3 — 前向传播",
                                "desc": "数据从输入层经过隐藏层流向输出层，每层通过加权求和与激活函数进行非线性变换",
                                "bg": "linear-gradient(90deg, #e3f2fd, #bbdefb)",
                                "title_color": "#1565c0",
                            },
                            1: {
                                "mode": "backward",
                                "title": "🔴 Step 2/3 — 反向传播",
                                "desc": "计算预测值与真实值的误差（损失），梯度从输出层向输入层逐层回传",
                                "bg": "linear-gradient(90deg, #ffebee, #ffcdd2)",
                                "title_color": "#c62828",
                            },
                            2: {
                                "mode": "weights",
                                "title": "🔵 Step 3/3 — 权重更新",
                                "desc": "根据梯度和学习率调整权重，使下一次预测更接近真实值",
                                "bg": "linear-gradient(90deg, #e8f5e9, #c8e6c9)",
                                "title_color": "#2e7d32",
                            },
                        }
                        current = phase_config.get(anim_phase, phase_config[2])

                        st.markdown(f"""
                        <div style="background: {current["bg"]}; padding: 12px 20px; border-radius: 8px; margin-bottom: 10px;">
                            <span style="font-size: 16px; font-weight: bold; color: {current["title_color"]};">{current["title"]}</span>
                            <span style="color: #555; margin-left: 10px;">{current["desc"]}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        fig = plot_network_structure(
                            st.session_state.nv_layer_sizes,
                            mode=current["mode"],
                            weights=anim_info.get("weights"),
                            activations=anim_info.get("activations"),
                            grads_W=anim_info.get("grads_W"),
                            figsize=(8, 5),
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        # 可视化模式切换
                        viz_mode = st.radio(
                            "显示模式",
                            ["weights", "forward", "backward", "structure"],
                            index=["weights", "forward", "backward", "structure"].index(st.session_state.nv_viz_mode),
                            key="nv_viz_mode_radio",
                            format_func=lambda m: {
                                "weights": "🔵 权重视图（正蓝负红）",
                                "forward": "🔷 前向传播（激活值）",
                                "backward": "🔴 反向传播（梯度）",
                                "structure": "⚪ 纯拓扑结构",
                            }[m],
                            horizontal=True,
                        )
                        st.session_state.nv_viz_mode = viz_mode

                        weights = None
                        activations = None
                        grads_W = None
                        if st.session_state.nv_last_info is not None:
                            weights = st.session_state.nv_last_info.get("weights")
                            activations = st.session_state.nv_last_info.get("activations")
                            grads_W = st.session_state.nv_last_info.get("grads_W")

                        fig = plot_network_structure(
                            st.session_state.nv_layer_sizes,
                            mode=viz_mode,
                            weights=weights,
                            activations=activations,
                            grads_W=grads_W,
                        )
                        st.pyplot(fig)
                        plt.close(fig)

                        st.markdown("""
                        <div style="font-size:12px; color:#666; margin-top:8px;">
                        <b>图例说明：</b><br>
                        • <b>权重视图</b>：正权重偏蓝，负权重偏红，连线粗细表示权重绝对值大小<br>
                        • <b>前向传播</b>：神经元亮度对应激活值大小（取第一个样本展示）<br>
                        • <b>反向传播</b>：神经元与连线颜色对应梯度绝对值大小（红色系）<br>
                        • <b>纯拓扑</b>：仅显示网络结构，不带数值映射
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"绘制网络结构时出错: {e}")
            else:
                st.info("构建网络后将在此显示神经网络拓扑结构，并支持切换前向/反向/权重视图。")

        # ==================== 实时状态 ====================
        st.markdown("---")
        st.markdown("<div class='sidebar-header'>📊 实时状态</div>", unsafe_allow_html=True)

        if st.session_state.nv_trainer is not None:
            trainer = st.session_state.nv_trainer
            hist = st.session_state.nv_history

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("当前轮次", trainer.current_epoch)
            with sc2:
                latest_loss = hist["loss"][-1] if hist["loss"] else "--"
                st.metric(
                    "最新损失",
                    f"{latest_loss:.6f}" if isinstance(latest_loss, (int, float)) else latest_loss
                )
            with sc3:
                if hist["acc"]:
                    st.metric("准确率", f"{hist['acc'][-1]:.4f}")
                else:
                    st.metric("准确率", "--")
        else:
            st.info("尚未构建网络，请配置参数后点击「构建网络」")

        # ==================== 原理讲解区 ====================
        st.markdown("---")
        st.markdown("<div class='sidebar-header'>📖 原理讲解</div>", unsafe_allow_html=True)

        if st.session_state.nv_trainer is not None:
            epoch = st.session_state.nv_trainer.current_epoch
            theory_text = _get_theory_text(epoch)
        else:
            theory_text = _get_theory_text(0)
        st.info(theory_text)

        # ==================== 运行日志 ====================
        st.markdown("---")
        with st.expander("📝 运行日志", expanded=False):
            if st.session_state.nv_logs:
                # 显示最近 30 条，倒序排列（最新的在上面）
                for log in reversed(st.session_state.nv_logs[-30:]):
                    st.text(log)
            else:
                st.caption("暂无日志，构建网络并开始训练后将在此显示。")

    # 三阶段单步动画：每次渲染一帧，等待后切换到下一帧
    if st.session_state.nv_anim_phase is not None:
        time.sleep(2.0)
        if st.session_state.nv_anim_phase < 2:
            st.session_state.nv_anim_phase += 1
        else:
            st.session_state.nv_anim_phase = None
            st.session_state.nv_anim_info = None
        st.rerun()


# =============================================================================
# 6. 训练控制函数
# =============================================================================

def _do_single_step():
    """执行单步训练，更新所有状态。"""
    trainer = st.session_state.nv_trainer
    hist = st.session_state.nv_history

    try:
        info = trainer.step()
        hist["epoch"].append(info["epoch"])
        hist["loss"].append(info["loss"])
        if info["accuracy"] is not None:
            hist["acc"].append(info["accuracy"])

        log_msg = f"[Step] Epoch {info['epoch']:04d} | Loss {info['loss']:.6f}"
        if info["accuracy"] is not None:
            log_msg += f" | Acc {info['accuracy']:.4f}"
        st.session_state.nv_logs.append(log_msg)
        if len(st.session_state.nv_logs) > 200:
            st.session_state.nv_logs = st.session_state.nv_logs[-200:]

        st.session_state.nv_last_info = info

        # 前4轮在日志中追加原理提示
        if info["epoch"] <= 4:
            theory = _get_theory_text(info["epoch"])
            st.session_state.nv_logs.append(f"[Theory] {theory.replace(chr(10), ' ')}")

        return info

    except Exception as e:
        st.error(f"单步训练时发生错误: {e}")
        return None


def _execute_auto_training():
    """执行自动训练循环，带进度条与实时状态更新。"""
    trainer = st.session_state.nv_trainer
    target_epochs = st.session_state.nv_target_epochs
    hist = st.session_state.nv_history

    if trainer is None or trainer.current_epoch >= target_epochs:
        return

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    update_interval = 5
    start_epoch = trainer.current_epoch

    try:
        while trainer.current_epoch < target_epochs:
            info = trainer.step()

            hist["epoch"].append(info["epoch"])
            hist["loss"].append(info["loss"])
            if info["accuracy"] is not None:
                hist["acc"].append(info["accuracy"])

            log_msg = f"[Auto] Epoch {info['epoch']:04d} | Loss {info['loss']:.6f}"
            if info["accuracy"] is not None:
                log_msg += f" | Acc {info['accuracy']:.4f}"
            st.session_state.nv_logs.append(log_msg)
            if len(st.session_state.nv_logs) > 200:
                st.session_state.nv_logs = st.session_state.nv_logs[-200:]

            st.session_state.nv_last_info = info

            current = trainer.current_epoch
            if (current - start_epoch) % update_interval == 0 or current == target_epochs:
                progress = min(current / target_epochs, 1.0)
                progress_bar.progress(progress)
                acc_str = f"{info['accuracy']:.4f}" if info["accuracy"] is not None else "N/A"
                status_text.text(f"Epoch {current} / {target_epochs}  |  Loss: {info['loss']:.6f}  |  Acc: {acc_str}")
                time.sleep(0.02)

        progress_bar.progress(1.0)
        status_text.text(f"✅ 训练完成！共 {target_epochs} 轮，最终 Loss: {hist['loss'][-1]:.6f}")
        time.sleep(0.3)

    except Exception as e:
        st.error(f"自动训练过程中发生错误: {e}")
