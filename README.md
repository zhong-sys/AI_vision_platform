# 智视 AI · 算法可视化学习平台

## 项目简介

本项目是一个基于 Streamlit 的人工智能算法可视化学习平台。用户可以在浏览器中调整参数、观察图表和训练过程，并通过交互式说明理解算法原理。当前工程以教学演示为目标，不提供用户账户、数据库或文件上传服务。

## 功能模块

- 机器学习：分类、回归、聚类
- 神经网络：基础前馈网络、CNN、RNN、Transformer 注意力机制
- 国产大模型：DeepSeek、智谱 GLM、通义千问、文心一言的对话体验
- 统一的页面标题、侧边栏导航、页脚状态和资源路径处理
- 教学增强：页面学习目标、实验步骤、确定性参数解释、参数预设和可选基础算法对比

## 技术架构

- **页面层**：Streamlit 单页应用，通过 `st.session_state.current_page` 保持现有客户端路由行为。
- **公共组件**：`components/` 提供顶部栏、侧边栏、页脚和页面标题骨架。
- **路由层**：`router.py` 按既有页面 key 延迟加载页面模块。
- **算法层**：`pages_modules/` 保存各页面的算法、数据、指标、可视化和教学文本实现。
- **配置与资源**：`config.py` 读取 Streamlit secrets，`utils/helpers.py` 统一解析静态资源路径，`assets/` 保存图片和字体。
- **启动与打包**：开发时直接运行 `main.py`；发布构建使用 `launcher.py` 和 `launcher.spec`。

## 项目结构

```text
AI_vision_platform/
├── main.py                    # Streamlit 入口：配置、状态、公共骨架和路由调用
├── router.py                  # 页面 key 到页面入口的路由
├── app_constants.py           # 页面 key、显示名和 LLM 路由元数据
├── config.py                  # 颜色常量和 API Key 读取
├── components/                # 页面骨架与 experiment_panel 教学组件
├── pages_modules/             # 页面 UI 与算法模块
├── utils/helpers.py           # resource_path() 等公共工具
├── assets/                    # logo、模型 logo、字体和演示图片
├── tests/                     # import、路由、资源和依赖图测试
├── scripts/smoke_test.py      # 快速页面模块导入检查
├── launcher.py                # PyInstaller 启动器
├── launcher.spec              # 发布构建配置
├── requirements.txt           # Python 依赖
└── .streamlit/               # 本地 Streamlit 配置与 secrets 示例
```

`pages_modules/NeuralVis/`、`.idea/`、旧备用入口和 Python 缓存等不参与当前 Streamlit 主程序的内容保存在本地被忽略目录 `.local_archive/phase3_legacy/`，不会随仓库提交。当前部署入口为 `main.py`；如需生成桌面版，可使用 `launcher.py` 与 `launcher.spec`。

## 安装方式

建议使用 Python 3.8 或更高版本，并在项目目录创建虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 启动方式

开发环境直接运行：

```bash
streamlit run main.py
```

默认地址为 `http://localhost:8501`。也可以运行 `python launcher.py`，该入口会以相同的 `main.py` 启动 Streamlit 并尝试打开浏览器。

## secrets 配置

复制 `.streamlit/secrets.toml.example` 为本机专用的 `.streamlit/secrets.toml`，再填写需要使用的提供商密钥：

```toml
DEEPSEEK_API_KEY = "你的 DeepSeek Key"
ZHIPU_API_KEY = "你的智谱 Key"
ALIYUN_API_KEY = "你的阿里云 Key"
BAIDU_API_KEY = "你的百度 Key"
```

应用通过 `config.get_api_key()` 从 `st.secrets` 读取密钥；未配置的模型会显示配置提示，不会发起 API 请求。不要把真实密钥写入源代码、日志或提交记录。

## 开发说明

- 保持既有页面 key、控件 key、参数默认值和 `session_state` key 不变。
- 算法实现与模型调用逻辑位于页面模块中，公共组件只负责展示和导航骨架。
- 修改页面后先执行语法检查、导入 smoke test，再用 Streamlit 手动验证目标路由。
- 快速检查正式页面入口：

  ```bash
  python scripts/smoke_test.py
  ```

- 运行单元测试：

  ```bash
  python -m unittest discover -s tests -v
  ```

## 生产部署说明

生产环境应使用固定版本的构建环境、进程管理器和反向代理来运行 Streamlit，例如：

```bash
streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```

TLS、域名、访问控制和进程重启策略应由已有的 Nginx/平台基础设施负责；本项目不新增认证、数据库或网络服务。部署时只复制源代码、依赖和静态资源，并通过部署平台的 secrets 机制提供 `.streamlit/secrets.toml` 内容。

## 安全注意事项

- `.streamlit/secrets.toml`、`.env*`、日志、缓存、虚拟环境和构建产物均已加入 `.gitignore`。
- 提交前使用 `git status` 检查，确认没有真实 API Key、Token 或本地配置文件。
- 不要把 API Key 放进页面文案、异常提示、截图或浏览器端 JavaScript。
- 保持 `.streamlit/config.toml` 的 CORS、XSRF 等安全配置，不要为排查问题临时关闭保护后直接部署。
- 生产环境仅通过受信任的反向代理暴露服务，并限制服务器端口的外部访问。
