import json
from typing import List, Dict, Any
from src import llm
from src.tools import stock_tools

class StockAgent:
    """Agent for stock analysis using Nemotron."""
    
    def __init__(self):
        self.llm_client = llm.create_client()
        self.conversation_history: List[Dict[str, str]] = []
        
        # Tool definitions for the LLM
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Get the current stock price for a given ticker symbol.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol (e.g., NVDA, AAPL)"
                            }
                        },
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_company_info",
                    "description": "Get summary information about a company.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol"
                            }
                        },
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plot_stock_history",
                    "description": "Plot the stock price history for a given period.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol"
                            },
                            "period": {
                                "type": "string",
                                "description": "The time period to plot (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)",
                                "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
                            }
                        },
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "simulate_investment",
                    "description": "Simulate what would happen if you invested a certain amount of money in a stock over a time period. Shows profit/loss and portfolio value over time.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol"
                            },
                            "amount": {
                                "type": "number",
                                "description": "The investment amount in dollars (e.g., 500, 1000)"
                            },
                            "period": {
                                "type": "string",
                                "description": "The time period to simulate (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)",
                                "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
                            }
                        },
                        "required": ["ticker", "amount"]
                    }
                }
            }
        ]
        
        # Map function names to actual functions
        self.available_functions = {
            "get_stock_price": stock_tools.get_stock_price,
            "get_company_info": stock_tools.get_company_info,
            "plot_stock_history": stock_tools.plot_stock_history,
            "simulate_investment": stock_tools.simulate_investment
        }
        
        self.system_message = """You are a helpful financial assistant powered by NVIDIA Nemotron.
        You can provide stock prices, company information, historical price plots, and investment simulations.
        
        When asked to plot or show a chart, ALWAYS use the plot_stock_history tool.
        
        When asked "What if I invested..." or similar simulation questions, use the simulate_investment tool.
        
        CRITICAL INSTRUCTION FOR PLOTTING:
        1. CAREFULLY ANALYZE the user's request for any time period mentions (e.g., "6 months", "5 years", "1 month", "YTD").
        2. Map these to the closest available period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        3. ONLY default to '1y' if the user did NOT specify a period.
        4. If the user asks for "over the years" without a number, use 'max' or '5y'.
        
        For investment simulations:
        - Extract the dollar amount from phrases like "$500", "500 dollars", etc.
        - Extract the time period (e.g., "1 year ago" = "1y", "6 months ago" = "6mo")
        - If no period is specified, default to '1y'
        
        Be concise and professional in your responses."""

    def chat(self, user_input: str) -> tuple[str, str | None]:
        """Process user input and return a response.
        
        Returns:
            tuple: (response_text, image_path or None)
        """
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Prepare messages for API (System + History)
        messages = [{"role": "system", "content": self.system_message}] + self.conversation_history
        
        # Track if we generated a plot
        plot_path = None
        
        try:
            # First call to LLM
            response = self.llm_client.chat(messages, tools=self.tool_definitions)
            response_message = response["choices"][0]["message"]
            
            # Check for tool calls
            tool_calls = response_message.get("tool_calls")
            
            if tool_calls:
                # Append assistant's message with tool calls to history
                messages.append(response_message)
                self.conversation_history.append(response_message)
                
                # Execute tool calls
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])
                    
                    function_to_call = self.available_functions.get(function_name)
                    if function_to_call:
                        function_response = function_to_call(**function_args)
                        
                        # Check if this was a plot function or simulation
                        if function_name in ["plot_stock_history", "simulate_investment"] and "PLOTLY_CHART:" in str(function_response):
                            # Extract the JSON
                            import re
                            match = re.search(r"PLOTLY_CHART:(.*?)(?:\n|$)", str(function_response), re.DOTALL)
                            if match:
                                plot_path = match.group(1).strip()
                                # Remove everything after the first newline or closing brace pattern
                                # The JSON ends when we hit **
                                if "**" in plot_path:
                                    plot_path = plot_path.split("**")[0].strip()
                        
                        # Append tool response to history
                        messages.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": str(function_response)
                        })
                        # Also update local history
                        self.conversation_history.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": str(function_response)
                        })
                
                # Second call to LLM to get final response
                second_response = self.llm_client.chat(messages)
                final_content = second_response["choices"][0]["message"]["content"]
                
                self.conversation_history.append({"role": "assistant", "content": final_content})
                return final_content, plot_path
            
            else:
                # No tool calls, just return content
                content = response_message["content"]
                self.conversation_history.append({"role": "assistant", "content": content})
                return content, None
                
        except Exception as e:
            return f"Error: {str(e)}", None

    def clear_history(self):
        self.conversation_history = []
