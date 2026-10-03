"""Start the MCP server over stdio like an agent would, list its tools and call two of them.

    python scripts/mcp_smoke.py data/olist
"""
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(db: str) -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "posterior.cli", "mcp", "--db", db],
                                   env=dict(os.environ))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])
            r = await session.call_tool("describe_database", {})
            print("entities:", json.loads(r.content[0].text)["entities"])
            r = await session.call_tool("formulate", {"question": "Which sellers will stop selling in the next 30 days?"})
            d = json.loads(r.content[0].text)
            print("top reading:", d["readings"][0]["description"], round(d["readings"][0]["probability"], 2))
            print("clarification:", d["clarification"])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "data/olist"))
