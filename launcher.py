# launcher.py
import logging
import os
import sys
import time
import threading
import webbrowser

from utils.helpers import resource_path


LOGGER = logging.getLogger(__name__)


def main():
    # 定位主程序文件（你的主程序是 main.py）
    internal_dir = resource_path(".")
    main_py = os.path.join(internal_dir, "main.py")

    if not os.path.isfile(main_py):
        # 避免使用 input，直接等待后退出（无控制台时不会报错）
        msg = "错误：找不到 main.py，程序即将退出..."
        print(msg)
        time.sleep(5)
        sys.exit(1)

    # 设置 Python 搜索路径，确保 pages_modules 等模块能被正确导入
    if internal_dir not in sys.path:
        sys.path.insert(0, internal_dir)

    # 配置 Streamlit 启动参数
    host = "127.0.0.1"
    port = "8501"
    url = f"http://{host}:{port}"

    print("=" * 56)
    print("       智视AI · 算法可视化学习平台")
    print("=" * 56)
    print("\n正在启动服务，请稍候...")
    print(f"服务地址: {url}")
    print("按 Ctrl+C 或关闭本窗口即可停止服务\n")

    # 延迟自动打开浏览器
    def open_browser():
        time.sleep(3)
        webbrowser.open(url, new=2)

    threading.Thread(target=open_browser, daemon=True).start()

    # 修改 sys.argv 以注入 Streamlit 启动命令
    sys.argv = [
        "streamlit", "run", main_py,
        "--server.headless", "true",
        "--server.address", host,
        "--server.port", port,
        "--global.developmentMode", "false",
        "--browser.gatherUsageStats", "false",
    ]

    try:
        from streamlit.web.cli import main as st_main
        st_main()
    except SystemExit:
        pass
    except Exception:
        LOGGER.exception("Streamlit service failed to start")
        print("\n错误：启动服务时发生异常，请查看服务日志。")
        time.sleep(5)
        sys.exit(1)

if __name__ == "__main__":
    main()
