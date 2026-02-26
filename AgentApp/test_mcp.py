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
        print(f"Error: input image not found -> {input_image}")
        return

    server_params = StdioServerParameters(
        command="python",
        args=["/home/jason/Auto-Image-Restoration/AgentApp/mcp_server.py"],
        # cwd="/home/jason/Auto-Image-Restoration/AgentApp",
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("Initializing MCP session...")
                await session.initialize()

                print("Fetching available tools...")
                tools = await session.list_tools()
                print("Available tools:")
                if not tools.tools:
                    print("  (No tools found! Please check mcp_server.py implementation)")
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description}")

                print("\nCalling image restoration tool...")
                result = await session.call_tool(
                    "restore_image",
                    arguments={"image_path": str(input_image.absolute())}
                )

                print("Tool call result:")
                for content in result.content:
                    if content.type == "text":
                        print("  Text:", content.text)
                    elif content.type == "image":
                        try:
                            img_data = base64.b64decode(content.data)
                            output_image.write_bytes(img_data)
                            print(f"Success! Restored image saved to: {output_image.absolute()}")
                            print(f"File size: {output_image.stat().st_size:,} bytes")
                        except Exception as e:
                            print("Failed to decode/save image:", e)
                            traceback.print_exc()

    except Exception as e:
        print("\nAn error occurred:")
        traceback.print_exc()
        print("\nPossible causes:")
        print("1. mcp_server.py failed to start (wrong path, missing dependencies, CUDA unavailable, etc.)")
        print("2. Subprocess does not correctly implement the MCP protocol")
        print("3. restore_image tool not found or argument mismatch")
        print("4. Pipe communication interrupted")

    finally:
        print("\nSession ended")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print("asyncio error:", e)



