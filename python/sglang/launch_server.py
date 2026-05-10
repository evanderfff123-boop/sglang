"""启动推理服务器。"""
import asyncio
import os
import sys
import warnings

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()


def run_server(server_args):
    """根据参数选择服务器运行模式（gRPC / Ray / 默认 HTTP）。"""
    if server_args.encoder_only:
        # 编码器分离部署模式：将多模态编码器独立为单独服务
        if server_args.grpc_mode:
            from sglang.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
    elif server_args.grpc_mode:
        # 纯 gRPC 模式（无 HTTP），用于高性能场景
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.use_ray:
        # Ray 分布式模式，用于多机多卡部署
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # 默认模式：标准 HTTP 服务器（FastAPI + Uvicorn）
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    from sglang.srt.plugins import load_plugins

    # 加载第三方插件。通过 setuptools entry_points 机制发现：
    #   1. 硬件平台插件（sglang.srt.platforms 组）— 注册自定义硬件后端
    #   2. 通用插件（sglang.srt.plugins 组）— 注入 hook、替换类等
    # 环境变量 SGLANG_PLUGINS 可限制只加载指定插件（逗号分隔）。
    # 必须在所有其他 import 之前执行，因为插件可能修改导入行为。
    load_plugins()

    # 解析命令行参数，构造 ServerArgs 对象
    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        # 确保退出时清理所有子进程
        kill_process_tree(os.getpid(), include_parent=False)
