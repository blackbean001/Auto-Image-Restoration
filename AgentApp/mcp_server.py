import asyncio
import base64
import os
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# Reuse existing modules from the Flask app
from agentic_api import init_invoke_dict, get_compiled_graph

import sys
import logging

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

print = lambda *args, **kwargs: __builtins__['print'](*args, **kwargs, file=sys.stderr)

server = Server("image-restoration")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="restore_image",
            description="Automatically evaluate and restore a degraded image. Supports denoising, deblurring, super-resolution, deraining, and more.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Local path to the image file on the server"
                    },
                    "image_base64": {
                        "type": "string",
                        "description": "Base64-encoded image data (mutually exclusive with image_path)"
                    },
                    "image_format": {
                        "type": "string",
                        "description": "Image format, e.g. png or jpg. Required when using image_base64.",
                        "default": "png"
                    }
                }
            }
        ),
        types.Tool(
            name="evaluate_image",
            description="Evaluate image quality and return a degradation analysis report without performing any restoration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                },
                "required": ["image_path"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent | types.ImageContent]:

    if name == "restore_image":
        # Resolve image path — accept either a local path or base64-encoded data
        image_path = arguments.get("image_path")
        if not image_path and arguments.get("image_base64"):
            image_path = _save_base64_image(
                arguments["image_base64"],
                arguments.get("image_format", "png")
            )

        if not image_path or not os.path.exists(image_path):
            return [types.TextContent(type="text", text=f"Error: image not found at {image_path}")]

        # Invoke the LangGraph pipeline directly — no changes needed to internal logic
        invoke_dict = init_invoke_dict(image_path)
        app_graph = get_compiled_graph()

        # app_graph.invoke() is synchronous and blocking; run it in a thread pool
        # to avoid blocking the asyncio event loop
        result = await asyncio.get_event_loop().run_in_executor(
            None, app_graph.invoke, invoke_dict
        )

        output_path = result.get("final_output_path", "")
        response = {
            "status": "success",
            "task_id": result.get("task_id", ""),
            "output_path": output_path,
            "initial_plan": result.get("initial_plan", []),
            "subtask_success": result.get("subtask_success", {})
        }

        # Return structured result metadata alongside the restored image
        contents = [types.TextContent(type="text", text=str(response))]

        # Attach the output image as an ImageContent block if it exists,
        # so the MCP host can render it directly without needing the file path
        if output_path and os.path.exists(output_path):
            contents.append(_load_image_content(output_path))

        return contents

    elif name == "evaluate_image":
        # Placeholder: wire this up to a standalone evaluation subgraph
        # once the LangGraph pipeline supports partial execution
        return [types.TextContent(type="text", text="evaluation-only mode: not yet implemented")]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


def _save_base64_image(b64_data: str, fmt: str) -> str:
    """Decode a base64 image and save it to the uploads directory."""
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    import uuid
    image_path = upload_dir / f"{uuid.uuid4()}.{fmt}"
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return str(image_path)


def _load_image_content(image_path: str) -> types.ImageContent:
    """Read an image file from disk and return it as a base64-encoded ImageContent block."""
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}
    mime_type = mime_map.get(suffix, "image/png")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return types.ImageContent(type="image", data=b64, mimeType=mime_type)


if __name__ == "__main__":
    from mcp.server.stdio import stdio_server

    async def run_server():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )

    asyncio.run(run_server())




