import os
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ==========================================
# 🛠️ THE TOOLS: DATA SENSORS & ACTIONS
# ==========================================

def inspect_csv_file(filename: str) -> str:
    """Reads a CSV file and returns its structure, columns, and sample row data."""
    try:
        if not os.path.exists(filename):
            return f"Error: The file '{filename}' does not exist."
        
        df = pd.read_csv(filename)
        info_summary = f"Columns Found: {list(df.columns)}\n"
        info_summary += f"Total Rows: {len(df)}\n"
        info_summary += f"Sample Raw Rows:\n{df.head(3).to_string()}\n"
        info_summary += f"Missing Values Summary:\n{df.isnull().sum().to_string()}"
        return info_summary
    except Exception as e:
        return f"Failed to inspect file. Error: {str(e)}"

def execute_cleaning_operation(python_code_block: str) -> str:
    """Executes a generated piece of Python cleaning code on the data directory safely."""
    try:
        # Pre-execution formatting strip
        clean_code = python_code_block.replace("```python", "").replace("```", "").strip()
        
        # Isolated execution memory space
        local_scope = {"pd": pd, "os": os}
        exec(clean_code, globals(), local_scope)
        return "Success: The custom data cleaning routine ran without exceptions."
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
                    "filename": {"type": "string", "description": "The name of the csv data file."}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_cleaning_operation",
            "description": "Executes a generated piece of Python cleaning code safely. Parameter must be clean executable python code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "python_code_block": {"type": "string", "description": "The pure executable python cleaning script block using pandas."}
                },
                "required": ["python_code_block"]
            }
        }
    }
]

# ==========================================
# 🧠 THE ENGINE
# ==========================================

def run_data_cleaning_agent():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    # 🎯 FIX 1: Updated to the live, supported 27B tool-use model
    model_name = "qwen/qwen3.8-27b"
    
    print("\n=============================================")
    print("🧹 UNLIMITED DATA CLEANING AGENT BOOTED 🧹")
    print("=============================================\n")
    
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an expert Data Cleaning Engineer. Your objective is to clean 'dirty_data.csv'.\n"
                "Step 1: Run inspect_csv_file to analyze what features are present.\n"
                "Step 2: Generate a standard python script block using pandas that cleans formatting issues. "
                "Ensure you handle column header whitespaces, strip text values, convert headers/text to lowercase, "
                "correct mixed date formats uniformly to YYYY-MM-DD, drop completely empty records, "
                "and finalize saving the data structure using: df.to_csv('clean_data.csv', index=False).\n"
                "Step 3: Instantly execute that code block using execute_cleaning_operation."
            )
        },
        {"role": "user", "content": "Analyze 'dirty_data.csv', write a cleaning workflow, and execute it."}
    ]
    
    for turn in range(4):
        print(f"🤖 [Data Agent] Analyzing dataset pipeline (Turn {turn + 1}/4)...")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=DATA_TOOLS_SCHEMA,
            tool_choice="auto",
            # 🎯 FIX 2: Restrict max tokens to stop background loops from blowing past the 1000 OTPM cap
            max_tokens=600 
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"⚡ [Agent Tool Call Requested]: '{name}'")
                
                if name == "inspect_csv_file":
                    output = inspect_csv_file(args.get("filename"))
                elif name == "execute_cleaning_operation":
                    output = execute_cleaning_operation(args.get("python_code_block"))
                else:
                    output = "Error: Unknown data tool."
                    
                print(f"📥 [Sensor Data Received]:\n{output}\n")
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": name,
                    "content": output
                })
        else:
            print(f"\n[Agent Final Report]:\n{response_message.content}")
            break
            
    if os.path.exists("clean_data.csv"):
        print("🔥 Success! The Agent used its tool and created 'clean_data.csv'!")
        print("\n=== CLEAN FILE CONTENTS ===")
        print(pd.read_csv("clean_data.csv").to_string())
    else:
        print("\n❌ Pipeline completed, but clean_data.csv was not generated.")




if __name__ == "__main__":
    run_data_cleaning_agent()
