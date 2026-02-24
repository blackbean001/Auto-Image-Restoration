# test_mcp_client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["/absolute/path/to/mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()

            # 查看有哪些可用工具
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # 调用 restore_image 工具（传本地路径）
            result = await session.call_tool(
                "restore_image",
                arguments={"image_path": "/home/user/images/degraded.jpg"}
            )

            # 处理返回结果
            for content in result.content:
                if content.type == "text":
                    print("Result metadata:", content.text)
                elif content.type == "image":
                    # 把返回的 base64 图像保存到本地
                    import base64
                    with open("restored_output.png", "wb") as f:
                        f.write(base64.b64decode(content.data))
                    print("Restored image saved to restored_output.png")

asyncio.run(main())
