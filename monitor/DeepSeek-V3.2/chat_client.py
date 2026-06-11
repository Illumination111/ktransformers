"""
交互式对话客户端，通过 OpenAI 兼容 API 与 sglang 服务通信。

用法：
    python chat_client.py                         # 默认连接 localhost:30000
    python chat_client.py --port 8000             # 指定端口
    python chat_client.py --max-tokens 512        # 设置最大输出 token 数
    python chat_client.py --host 192.168.1.1      # 连接远程服务器
    python chat_client.py --no-stream             # 禁用流式输出

会话内可用命令：
    /tokens <N>     动态修改最大输出 token 数（例：/tokens 1024）
    /clear          清空对话历史，开始新对话
    /history        查看当前对话历史
    /system <text>  修改 system prompt
    /quit 或 /exit  退出
"""

import argparse
import json
import sys

try:
    import requests
except ImportError:
    print("请安装 requests：pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="sglang 交互式对话客户端")
    parser.add_argument("--host", default="127.0.0.1", help="服务地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=30000, help="服务端口（默认 30000）")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大输出 token 数（默认 512）")
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="系统提示词（默认：You are a helpful assistant.）",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="采样温度（默认 0.6）"
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="禁用流式输出，等待完整响应后再显示"
    )
    parser.add_argument(
        "--model", default="DeepSeek-V3.2", help="模型名称（默认 Qwen3-30B-A3B）"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# API 调用
# ---------------------------------------------------------------------------

def chat_stream(base_url, model, messages, max_tokens, temperature):
    """流式调用 /v1/chat/completions，逐 token 打印。返回完整回复文本。"""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            full_text += content
                        # 尝试获取 usage 信息（部分实现会在最后一条带上）
                        usage = data.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)
                    except json.JSONDecodeError:
                        pass
    except requests.exceptions.ConnectionError:
        print(f"\n[错误] 无法连接到 {base_url}，请确认服务已启动。")
        return None, 0, 0
    except requests.exceptions.HTTPError as e:
        print(f"\n[错误] HTTP {e.response.status_code}: {e.response.text}")
        return None, 0, 0

    print()  # 换行
    return full_text, prompt_tokens, completion_tokens


def chat_no_stream(base_url, model, messages, max_tokens, temperature):
    """非流式调用，等待完整响应。"""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        print(content)
        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    except requests.exceptions.ConnectionError:
        print(f"\n[错误] 无法连接到 {base_url}，请确认服务已启动。")
        return None, 0, 0
    except requests.exceptions.HTTPError as e:
        print(f"\n[错误] HTTP {e.response.status_code}: {e.response.text}")
        return None, 0, 0


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def print_help():
    print(
        "\n可用命令：\n"
        "  /tokens <N>     修改最大输出 token 数（例：/tokens 1024）\n"
        "  /clear          清空对话历史\n"
        "  /history        查看当前对话历史\n"
        "  /system <text>  修改 system prompt\n"
        "  /help           显示此帮助\n"
        "  /quit 或 /exit  退出\n"
    )


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    # 检查服务连通性（用 /v1/models 端点，始终可用）
    # 注意：sglang 的 /health 在推理繁忙时会返回 503，属正常现象，不用作连通判断
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        model_ids = [m["id"] for m in models]
        print(f"[服务就绪] 可用模型: {model_ids}")
    except requests.exceptions.ConnectionError:
        print(f"[错误] 无法连接到 {base_url}，请确认 sglang 服务已启动。")
        sys.exit(1)
    except Exception as e:
        print(f"[警告] 连通性检查异常: {e}，继续尝试...")

    max_tokens = args.max_tokens
    system_prompt = args.system
    use_stream = not args.no_stream
    history = []  # [{"role": "...", "content": "..."}]

    print(f"\n{'='*60}")
    print(f"  sglang 交互式客户端")
    print(f"  服务地址: {base_url}")
    print(f"  模型: {args.model}")
    print(f"  最大输出 token: {max_tokens}  |  温度: {args.temperature}")
    print(f"  流式输出: {'是' if use_stream else '否'}")
    print(f"  输入 /help 查看命令")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出]")
            break

        if not user_input:
            continue

        # 处理内部命令
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd in ("/quit", "/exit"):
                print("[退出]")
                break

            elif cmd == "/clear":
                history.clear()
                print("[对话历史已清空]")
                continue

            elif cmd == "/history":
                if not history:
                    print("[对话历史为空]")
                else:
                    for i, msg in enumerate(history):
                        role_label = "You" if msg["role"] == "user" else "Assistant"
                        content_preview = msg["content"][:100].replace("\n", " ")
                        print(f"  [{i+1}] {role_label}: {content_preview}{'...' if len(msg['content']) > 100 else ''}")
                continue

            elif cmd == "/tokens":
                if len(parts) < 2 or not parts[1].isdigit():
                    print("[用法] /tokens <正整数>，例：/tokens 1024")
                else:
                    max_tokens = int(parts[1])
                    print(f"[已设置] 最大输出 token 数 = {max_tokens}")
                continue

            elif cmd == "/system":
                if len(parts) < 2:
                    print(f"[当前 system prompt] {system_prompt}")
                else:
                    system_prompt = parts[1]
                    history.clear()
                    print(f"[已更新 system prompt，对话历史已清空]")
                continue

            elif cmd == "/help":
                print_help()
                continue

            else:
                print(f"[未知命令] {cmd}，输入 /help 查看可用命令")
                continue

        # 构造消息列表
        messages = [{"role": "system", "content": system_prompt}] + history + [
            {"role": "user", "content": user_input}
        ]

        print(f"\nAssistant: ", end="", flush=True)

        if use_stream:
            reply, p_tokens, c_tokens = chat_stream(
                base_url, args.model, messages, max_tokens, args.temperature
            )
        else:
            reply, p_tokens, c_tokens = chat_no_stream(
                base_url, args.model, messages, max_tokens, args.temperature
            )

        if reply is None:
            continue

        # 打印 token 统计
        if p_tokens or c_tokens:
            print(f"  [token 统计] 输入: {p_tokens}  输出: {c_tokens}  合计: {p_tokens + c_tokens}")
        else:
            print(f"  [输出已截断 max_tokens={max_tokens}，如需更长回复请用 /tokens <N> 调整]"
                  if len(reply.split()) >= max_tokens * 0.9 else "")

        # 将本轮对话加入历史
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
