from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators
from tradingagents.agents.utils.logging_utils import log_report_preview
from tradingagents.agents.utils.tool_runner import (
    extract_text_from_content,
    run_chain_with_tools,
)
from tradingagents.prompt_registry import prompt_text


def create_market_analyst(llm):
    prompt_name = "market_analyst"

    def market_analyst_node(state):
        scheduled_list = state.get("scheduled_analysts", []) or []
        scheduled_plan_list = state.get("scheduled_analysts_plan", []) or []
        scheduled_plan = {item.lower() for item in scheduled_plan_list}
        action = state.get("orchestrator_action", "").lower()
        ticker = state.get("target_ticker") or state["company_of_interest"]
        if (scheduled_plan and "market" not in scheduled_plan) or action not in ("", "monitor", "escalate", "trade", "execute"):
            try:
                print(f"[Market Analyst] Skipping for ticker {ticker} | action={action} | scheduled_plan={scheduled_plan}")
            except Exception:
                pass
            return {
                "market_report": "Market analyst skipped by orchestrator directive.",
                "scheduled_analysts": [item for item in scheduled_list if item.lower() != "market"],
                "scheduled_analysts_plan": scheduled_plan_list,
            }
        current_date = state["trade_date"]
        company_name = ticker

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = prompt_text(prompt_name)
        if not system_message:
            raise RuntimeError(
                f"Missing prompt configuration for {prompt_name}. "
                "Ensure prompts/market_analyst.json exists with system_prompt text."
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        try:
            print(f"[Market Analyst] Running analysis for {ticker} | action={action} | scheduled={scheduled_list}")
        except Exception:
            pass

        try:
            result, message_history, tool_runs = run_chain_with_tools(
                chain,
                tools,
                state["messages"],
            )
        except Exception as exc:
            report = f"Market analyst failed: {exc}"
            updated_schedule = [item for item in scheduled_list if item.lower() != "market"]
            return {
                "messages": state["messages"],
                "market_report": report,
                "scheduled_analysts": updated_schedule,
                "scheduled_analysts_plan": scheduled_plan_list,
            }

        report = extract_text_from_content(getattr(result, "content", "")) or "Market analyst produced no narrative."
        updated_schedule = [item for item in scheduled_list if item.lower() != "market"]

        payload = {
            "messages": message_history,
            "market_report": report,
            "scheduled_analysts": updated_schedule,
            "scheduled_analysts_plan": scheduled_plan_list,
        }
        log_report_preview("Market Analyst", ticker, report)
        try:
            print(
                f"[Market Analyst] Completed step for {ticker} | tool_runs={tool_runs} | report_len={len(report) if report else 0}"
            )
        except Exception:
            pass

        return payload

    return market_analyst_node
