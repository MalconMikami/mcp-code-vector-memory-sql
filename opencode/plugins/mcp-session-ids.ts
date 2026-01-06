import type { Plugin } from "@opencode-ai/plugin";

const plugin: Plugin = async () => ({
  "tool.execute.before": async (input, output) => {
    if (!input.tool.startsWith("mcp-code-vector-memory-sql_")) return;
    if (!output.args || typeof output.args !== "object") return;

    const current = (output.args as any).session_id;
    if (!current || current === "default" || current === "opencode-default") {
      output.args.session_id = input.sessionID;
    }
  },
});

export default plugin;

