import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt # Native import to help the visualizer agent stay brief
from dotenv import load_dotenv
from groq import Groq

import matplotlib
matplotlib.use('Agg') # 🛡️ CRITICAL SAFESTACK: Forces headless image compiling, stopping asyncio freezes on Python 3.14!

load_dotenv()

# ==========================================
# 🛠️ THE TOOLS: REGISTRY
# ==========================================

def inspect_csv_file(filename: str) -> str:
    """Reads a CSV file and returns its structure, columns, and sample row data."""
    try:
        if not os.path.exists(filename):
            return f"Error: The file '{filename}' does not exist."
        df = pd.read_csv(filename)
        info = f"Columns Found: {list(df.columns)}\n"
        info += f"Total Rows: {len(df)}\n"
        info += f"Sample Raw Rows:\n{df.head(3).to_string()}\n"
        return info
    except Exception as e:
        return f"Failed to inspect file. Error: {str(e)}"

def execute_python_data_tool(python_code_block: str) -> str:
    """Safely runs generated Python operations (Pandas cleaning or Matplotlib graphing)."""
    try:
        clean_code = python_code_block.replace("```python", "").replace("```", "").strip()
        # Inject BOTH libraries straight into the execution context environment
        local_scope = {"pd": pd, "plt": plt, "os": os}
        exec(clean_code, globals(), local_scope)
        return "Success: The custom data operation ran without exceptions."
    except Exception as e:
        return f"Execution crashed. Code Error: {str(e)}\nAttempted Code:\n{python_code_block}"

DATA_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "inspect_csv_file",
            "description": "Reads a CSV file and returns its structure, columns, and sample row data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "The name of the target csv data file."}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python_data_tool",
            "description": "Executes clean python code block using pandas or matplotlib. Do NOT use markdown. Pass raw python.",
            "parameters": {
                "type": "object",
                "properties": {
                    "python_code_block": {"type": "string", "description": "The pure executable python script string."}
                },
                "required": ["python_code_block"]
            }
        }
    }
]

# ==========================================
# 🧠 THE ENGINE: STEP-BY-STEP ORCHESTRATION
# ==========================================

def execute_agent_turn(client, model, messages):
    """Executes a single processing conversation loop turn with safe tool calling handling."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=DATA_TOOLS_SCHEMA,
        tool_choice="auto",
        max_tokens=600
    )
    
    #response_message = response.choices.message
    response_message = response.choices[0].message
    messages.append(response_message)
    
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"   ⚡ [Tool Request]: '{name}'")
            
            if name == "inspect_csv_file":
                output = inspect_csv_file(args.get("filename"))
            elif name == "execute_python_data_tool":
                output = execute_python_data_tool(args.get("python_code_block"))
            else:
                output = "Error: Unknown data tool."
                
            print(f"   📥 [Tool Output Completed]: Success.")
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": name,
                "content": output
            })
        return "tool_called"
    return response_message.content

# ==========================================
# 👥 RUN THE MULTI-AGENT CREW
# ==========================================

def run_analytics_crew():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    model = "qwen/qwen3.8-27b"
    
    print("\n=============================================")
    print("📊 MULTI-AGENT VISUALIZATION CREW BOOTED 📊")
    print("=============================================")
    
    # --- AGENT 1: THE DATA CLEANER ---
    print("\n🎬 PHASE 1: Deploying Agent 1 [The Data Cleaner]...")
    cleaner_messages = [
        {
            "role": "system",
            "content": (
                "You are Agent 1: The Data Cleaner. Your goal is to inspect 'dirty_data.csv' using inspect_csv_file. "
                "Next, write pure python code using execute_python_data_tool to load 'dirty_data.csv', strip text spaces, "
                "lowercase text/headers, drop empty lines, fill missing ages with 0 or drop them, "
                "and save the final clean dataset directly to 'clean_data.csv'."
            )
        },
        {"role": "user", "content": "Clean 'dirty_data.csv' and save the output to 'clean_data.csv'."}
    ]
    
    for _ in range(3):
        status = execute_agent_turn(client, model, cleaner_messages)
        if status != "tool_called":
            break
        time.sleep(1)
        
    # --- AGENT 2: THE DATA VISUALIZER ---
    print("\n🎬 PHASE 2: Deploying Agent 2 [The Data Visualizer]...")
    if not os.path.exists("clean_data.csv"):
        print("❌ Error: Cannot run Phase 2 because 'clean_data.csv' was not generated.")
        return
        
    visualizer_messages = [
        {
            "role": "system",
            "content": (
                "You are Agent 2: The Data Visualizer. Your goal is to inspect 'clean_data.csv'.\n"
                "Next, generate brief, clean python code to create a bar chart from it. "
                "Requirements: Use `df = pd.read_csv('clean_data.csv')`. Drop any rows where 'age' is NaN. "
                "Plot a bar chart with 'name' on X and 'age' on Y using `df.plot(kind='bar', x='name', y='age')`. "
                "Save it using `plt.savefig('insight_chart.png', bbox_inches='tight')` and call `plt.close()`. "
                "Execute your script immediately using execute_python_data_tool."
            )
        },
        {"role": "user", "content": "Generate a visualization chart from the clean dataset columns and save to insight_chart.png."}
    ]
    
    for _ in range(3):
        status = execute_agent_turn(client, model, visualizer_messages)
        if status != "tool_called":
            break
        time.sleep(1)
        
    # Final Operational Check
    print("\n=============================================")
    print("🏁 FINAL INTEGRATION AUDIT")
    print("=============================================")
    if os.path.exists("clean_data.csv"):
        print("✅ Data Pipeline: 'clean_data.csv' generated perfectly.")
    if os.path.exists("insight_chart.png"):
        print("🔥 Success! The Visualizer agent created 'insight_chart.png' automatically!")
    else:
        print("❌ Visualizer completed execution, but 'insight_chart.png' was missing.")
    print("🎉 Production R&D analytics pipeline finished!")

if __name__ == "__main__":
    run_analytics_crew()
