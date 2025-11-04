from fastmcp import FastMCP

from .tools import forecast_tool


# Create the FastMCP server instance
mcp = FastMCP(
    name="tsfm_mcp_server",
    version="1.0.0",
    description="MCP server for TSFM capabilities.",
)


mcp.add_tool(forecast_tool)


if __name__ == "__main__":
    # Start the MCP server
    mcp.run()
