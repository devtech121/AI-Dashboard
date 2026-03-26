import os
import pandas as pd
from modules.chat_engine import ChatEngine
from modules.profiler import DataProfiler


def run_eval():
    path = os.path.join("sample_data", "ecommerce.csv")
    df = pd.read_csv(path)
    profile = DataProfiler().profile(df)
    chat = ChatEngine()

    questions = [
        "Provide me all information about order 1044",
        "Provide me all information about order_id = 1077",
        "What is the average revenue by region?",
        "Top 5 products by sales",
        "Orders in the last 30 days",
    ]

    print("Chat Eval")
    print("-" * 40)
    for q in questions:
        out = chat.ask(q, df, profile)
        ok = out.get("result") is not None or "Could not generate code" not in out.get("answer", "")
        print(f"Q: {q}")
        print(f"OK: {ok} | Answer: {out.get('answer')}")
        print("-" * 40)


if __name__ == "__main__":
    run_eval()
