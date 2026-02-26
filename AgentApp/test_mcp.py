import asyncio
import base64
import traceback
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    input_image = Path("/home/jason/Auto-Image-Restoration/AgentApp/demo_input/input.png")
    output_image = Path("restored_output.png")

    if not input_image.is_file():
        print(f"错误：输入图片不存在 -> {input_image}")
        return

    server_params = StdioServerParameters(
        command="python",
        args=["/home/jason/Auto-Image-Restoration/AgentApp/mcp_server.py"],
        # 如果你的 mcp_server.py 需要特定环境变量或 cwd，可以在这里加
        # cwd="/home/jason/Auto-Image-Restoration/AgentApp",
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("正在初始化 MCP 会话...")
                await session.initialize()

                print("正在获取可用工具列表...")
                tools = await session.list_tools()
                print("可用工具：")
                if not tools.tools:
                    print("（没有找到任何工具！请检查 mcp_server.py 是否正确实现）")
                for tool in tools.tools:
                    print(f" - {tool.name}: {tool.description}")

                print("\n正在调用图像修复工具...")
                result = await session.call_tool(
                    "restore_image",
                    arguments={"image_path": str(input_image.absolute())}
                )

                print("工具调用返回结果：")
                for content in result.content:
                    if content.type == "text":
                        print("  文本内容:", content.text)
                    elif content.type == "image":
                        try:
                            img_data = base64.b64decode(content.data)
                            output_image.write_bytes(img_data)
                            print(f"成功！修复后的图片已保存到：{output_image.absolute()}")
                            print(f"文件大小：{output_image.stat().st_size:,} 字节")
                        except Exception as e:
                            print("解码/保存图片失败:", e)
                            traceback.print_exc()

    except Exception as e:
        print("\n发生错误：")
        traceback.print_exc()
        print("\n可能的原因：")
        print("1. mcp_server.py 启动失败（路径错、缺少依赖、CUDA不可用等）")
        print("2. 子进程没有正确实现 MCP 协议")
        print("3. restore_image 工具不存在或参数不匹配")
        print("4. 管道通信中断")

    finally:
        print("\n会话已结束")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n用户中断 (Ctrl+C)")
    except Exception as e:
        print("asyncio error: ", e)

