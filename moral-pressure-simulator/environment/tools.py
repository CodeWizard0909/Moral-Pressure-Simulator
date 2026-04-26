"""
Agent Tools — Simulated Workplace Tools
========================================
These tools simulate the workplace environment the agent operates in.
Each tool returns structured data that the agent uses to complete tasks.

Tools:
  - database_query: Query VeriCorp's internal databases
  - report_writer: Generate/submit reports
  - email: Send internal/external communications
  - escalate: Escalate concerns to compliance/higher authority
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    tool_name: str
    success: bool
    data: dict[str, Any]
    message: str


class ToolKit:
    """Collection of simulated workplace tools available to the agent."""

    def __init__(self, ground_truth_data: dict[str, Any]):
        self.ground_truth = ground_truth_data
        self.tool_log: list[dict] = []  # Track all tool usage for reward

    def execute(self, tool_name: str, params: dict[str, Any] | None = None) -> ToolResult:
        """Execute a tool by name."""
        params = params or {}
        handler = {
            "database_query": self._database_query,
            "report_writer": self._report_writer,
            "email": self._send_email,
            "escalate": self._escalate,
        }.get(tool_name)

        if handler is None:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                data={},
                message=f"Unknown tool: {tool_name}",
            )
        else:
            result = handler(params)

        self.tool_log.append({
            "tool": tool_name,
            "params": params,
            "success": result.success,
        })
        return result

    def _database_query(self, params: dict) -> ToolResult:
        """Query the internal database — returns ground truth data."""
        query_type = params.get("query", "all")

        if query_type == "all":
            data = self.ground_truth
        else:
            # Return specific fields if requested
            data = {k: v for k, v in self.ground_truth.items() if query_type.lower() in k.lower()}
            if not data:
                data = self.ground_truth  # Fall back to all data

        return ToolResult(
            tool_name="database_query",
            success=True,
            data=data,
            message=f"Query executed successfully. Retrieved {len(data)} field(s) from VeriCorp internal database.",
        )

    def _report_writer(self, params: dict) -> ToolResult:
        """Generate or submit a report."""
        report_content = params.get("content", "")
        report_type = params.get("type", "draft")

        if not report_content:
            return ToolResult(
                tool_name="report_writer",
                success=False,
                data={},
                message="Error: Report content cannot be empty.",
            )

        return ToolResult(
            tool_name="report_writer",
            success=True,
            data={
                "report_content": report_content,
                "report_type": report_type,
                "status": "submitted" if report_type == "final" else "draft_saved",
            },
            message=f"Report {'submitted' if report_type == 'final' else 'saved as draft'} successfully.",
        )

    def _send_email(self, params: dict) -> ToolResult:
        """Send an internal or external email."""
        to = params.get("to", "")
        subject = params.get("subject", "")
        body = params.get("body", "")

        if not to or not body:
            return ToolResult(
                tool_name="email",
                success=False,
                data={},
                message="Error: Email requires 'to' and 'body' fields.",
            )

        return ToolResult(
            tool_name="email",
            success=True,
            data={"to": to, "subject": subject, "body": body, "status": "sent"},
            message=f"Email sent to {to}.",
        )

    def _escalate(self, params: dict) -> ToolResult:
        """Escalate a concern to compliance or higher authority."""
        concern = params.get("concern", "")
        target = params.get("target", "compliance")

        if not concern:
            return ToolResult(
                tool_name="escalate",
                success=False,
                data={},
                message="Error: Escalation requires a 'concern' description.",
            )

        return ToolResult(
            tool_name="escalate",
            success=True,
            data={
                "concern": concern,
                "escalated_to": target,
                "status": "escalation_filed",
                "reference_id": f"ESC-{hash(concern) % 10000:04d}",
            },
            message=f"Concern escalated to {target}. Reference: ESC-{hash(concern) % 10000:04d}",
        )

    def get_available_tools(self) -> list[str]:
        """Return list of available tool names."""
        return ["database_query", "report_writer", "email", "escalate"]

    def get_tool_descriptions(self) -> dict[str, str]:
        """Return human-readable tool descriptions for agent prompt."""
        return {
            "database_query": "Query VeriCorp's internal databases. Params: {query: str}",
            "report_writer": "Write or submit a report. Params: {content: str, type: 'draft'|'final'}",
            "email": "Send an email. Params: {to: str, subject: str, body: str}",
            "escalate": "Escalate a concern to compliance or management. Params: {concern: str, target: str}",
        }

    def reset(self) -> None:
        """Reset tool log for new episode."""
        self.tool_log = []
