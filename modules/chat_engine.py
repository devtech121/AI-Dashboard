"""
Chat Engine Module
Implements Option B: LLM generates pandas code → safe execution → return result
"""

import logging
import re
import traceback
import pandas as pd
import numpy as np
from utils.config import AppConfig

logger = logging.getLogger("ai_dashboard.chat_engine")


class ChatEngine:
    """
    Chat-with-data engine.
    LLM generates pandas code → sandboxed execution → result returned to user.
    """

    FORBIDDEN_TOKENS = [
        "__import__", "exec(", "eval(", "open(", "os.", "sys.", "subprocess",
        "shutil", "pathlib", "__builtins__", "globals(", "locals(",
        "getattr", "setattr", "delattr", "compile(", "breakpoint",
        "input(", "print(", "__class__", "__bases__", "__subclasses__",
    ]

    def ask(self, question: str, df: pd.DataFrame, profile: dict) -> dict:
        """
        Answer a question about the data.
        
        Returns dict with:
          answer, code (optional), result (optional)
        """
        # Try AI code generation first
        code = self._generate_code(question, df, profile)

        if code:
            execution_result = self._safe_execute(code, df)
            if execution_result["success"]:
                answer = self._format_answer(question, execution_result["result"], code)
                return {
                    "answer": answer,
                    "code": code,
                    "result": execution_result["result"]
                }
            else:
                # Code execution failed — try rule-based fallback
                logger.warning(f"Code execution failed: {execution_result['error']}")
                fallback = self._rule_based_answer(question, df, profile)
                if fallback:
                    return fallback

        # Final fallback — general response
        return self._general_fallback(question, df, profile)

    # ── Code Generation ────────────────────────────────────────

    def _generate_code(self, question: str, df: pd.DataFrame, profile: dict) -> str | None:
        """Generate pandas code using LLM."""
        try:
            from modules.ai_engine import AIEngine
            ai = AIEngine()

            # Build column context
            col_info = []
            for col in profile.get("columns", [])[:15]:
                info = f"{col['name']} ({col['type']}, {col['dtype']})"
                if col["type"] == "categorical" and col.get("top_values"):
                    top = list(col["top_values"].keys())[:3]
                    info += f" — values: {top}"
                col_info.append(info)

            cols_text = "\n".join([f"- {c}" for c in col_info])

            prompt = f"""You are a Python/pandas expert. Generate ONLY executable pandas code to answer the question.

DataFrame variable: df
Shape: {df.shape[0]} rows × {df.shape[1]} columns

Columns:
{cols_text}

Question: {question}

Rules:
1. Return ONLY the Python code, no explanation, no markdown, no comments
2. Store the final result in a variable named `result`
3. Use only pandas (pd) and numpy (np) operations
4. Do NOT use print(), display(), or plt
5. Keep code simple and safe
6. result should be a DataFrame, Series, scalar value, or string
7. For aggregations, always use .reset_index() on groupby results

Code:"""

            response = ai.call(
                prompt=prompt,
                max_tokens=256,
                temperature=0.05
            )

            if not response:
                return None

            code = self._extract_code(response)
            return code if code else None

        except Exception as e:
            logger.error(f"Code generation error: {e}")
            return None

    def _extract_code(self, response: str) -> str | None:
        """Extract Python code from LLM response."""
        if not response:
            return None

        # Remove markdown code blocks
        patterns = [
            r"```python\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If response looks like raw code
        lines = response.strip().split("\n")
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "result" in response:
                code_lines.append(line)

        if code_lines and "result" in "\n".join(code_lines):
            return "\n".join(code_lines)

        # Just return cleaned response if it contains result assignment
        cleaned = response.strip()
        if "result" in cleaned and "=" in cleaned:
            return cleaned

        return None

    # ── Safe Execution ─────────────────────────────────────────

    def _safe_execute(self, code: str, df: pd.DataFrame) -> dict:
        """
        Execute pandas code in a sandboxed environment.
        
        Returns dict with success, result, error.
        """
        # Security check
        security_check = self._check_code_safety(code)
        if not security_check["safe"]:
            return {
                "success": False,
                "result": None,
                "error": f"Code rejected for security: {security_check['reason']}"
            }

        # Build safe execution context
        safe_globals = {
            "__builtins__": {
                "len": len, "range": range, "list": list, "dict": dict,
                "str": str, "int": int, "float": float, "bool": bool,
                "tuple": tuple, "set": set, "zip": zip, "enumerate": enumerate,
                "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
                "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
                "isinstance": isinstance, "type": type,
            },
            "pd": pd,
            "np": np,
            "df": df.copy(),
        }

        try:
            exec(code, safe_globals)
            result = safe_globals.get("result", None)
            return {"success": True, "result": result, "error": None}
        except Exception as e:
            logger.warning(f"Code execution error: {e}\nCode:\n{code}")
            return {"success": False, "result": None, "error": str(e)}

    def _check_code_safety(self, code: str) -> dict:
        """Check code for potentially dangerous operations."""
        code_lower = code.lower()
        for token in self.FORBIDDEN_TOKENS:
            if token.lower() in code_lower:
                return {"safe": False, "reason": f"Forbidden operation: {token}"}

        # Check for allowed operations only
        allowed = set(AppConfig.SAFE_PANDAS_OPS)
        # Simple heuristic: reject if line has unusual patterns
        for line in code.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Allow assignments, method chains, comparisons
            if any(bad in stripped for bad in ["__", "import ", "exec(", "eval("]):
                return {"safe": False, "reason": f"Suspicious pattern: {stripped[:50]}"}

        return {"safe": True, "reason": None}

    # ── Answer Formatting ──────────────────────────────────────

    def _format_answer(self, question: str, result, code: str) -> str:
        """Format the execution result into a natural language answer."""
        if result is None:
            return "The query ran successfully but returned no results."

        if isinstance(result, pd.DataFrame):
            if len(result) == 0:
                return "No matching records found for your query."
            return (
                f"Here are the results for your query. "
                f"The table shows {len(result):,} row(s) and {len(result.columns)} column(s)."
            )

        if isinstance(result, pd.Series):
            if len(result) == 0:
                return "No data found."
            if len(result) <= 10:
                items = "\n".join([f"• **{k}**: {v:,.2f}" if isinstance(v, float) else f"• **{k}**: {v}" for k, v in result.items()])
                return f"Here are the results:\n\n{items}"
            return f"Found {len(result)} items. Displaying the data below."

        if isinstance(result, (int, float, np.integer, np.floating)):
            formatted = f"{result:,.4f}" if isinstance(result, float) else f"{result:,}"
            return f"**Answer:** {formatted}"

        if isinstance(result, str):
            return result

        return f"**Result:** {str(result)}"

    # ── Fallback Methods ───────────────────────────────────────

    def _rule_based_answer(self, question: str, df: pd.DataFrame, profile: dict) -> dict | None:
        """Rule-based Q&A for common question patterns."""
        q_lower = question.lower()

        try:
            # "how many rows" / "count"
            if any(phrase in q_lower for phrase in ["how many rows", "row count", "total records", "count of"]):
                return {
                    "answer": f"The dataset contains **{len(df):,} rows** in total.",
                    "code": "result = len(df)",
                    "result": len(df)
                }

            # "what columns" / "list columns"
            if any(phrase in q_lower for phrase in ["what columns", "list columns", "column names", "what are the"]):
                cols = ", ".join(df.columns.tolist())
                return {
                    "answer": f"The dataset has {len(df.columns)} columns: **{cols}**",
                    "code": "result = df.columns.tolist()",
                    "result": None
                }

            # "average" / "mean"
            if "average" in q_lower or "mean" in q_lower:
                num_cols = profile.get("numerical_cols", [])
                if num_cols:
                    col = num_cols[0]
                    val = df[col].mean()
                    return {
                        "answer": f"The average of **{col}** is **{val:,.2f}**",
                        "code": f"result = df['{col}'].mean()",
                        "result": val
                    }

            # "missing" / "null"
            if "missing" in q_lower or "null" in q_lower:
                missing = df.isnull().sum()
                missing = missing[missing > 0]
                if len(missing) == 0:
                    return {
                        "answer": "✅ Great news! There are **no missing values** in this dataset.",
                        "code": "result = df.isnull().sum()[df.isnull().sum() > 0]",
                        "result": None
                    }
                result_df = missing.reset_index()
                result_df.columns = ["Column", "Missing Count"]
                result_df["Missing %"] = (missing.values / len(df) * 100).round(2)
                return {
                    "answer": f"Found **{len(missing)} columns** with missing values.",
                    "code": "result = df.isnull().sum()[df.isnull().sum() > 0].reset_index()",
                    "result": result_df
                }

        except Exception as e:
            logger.warning(f"Rule-based fallback error: {e}")

        return None

    def _general_fallback(self, question: str, df: pd.DataFrame, profile: dict) -> dict:
        """Generic fallback response."""
        num_cols = profile.get("numerical_cols", [])
        cat_cols = profile.get("categorical_cols", [])

        answer = (
            f"I couldn't execute specific code for that question. "
            f"Your dataset has {len(df):,} rows, {len(num_cols)} numerical columns "
            f"({', '.join(num_cols[:3])}{'...' if len(num_cols) > 3 else ''}) and "
            f"{len(cat_cols)} categorical columns. "
            f"Try questions like: 'What is the average {num_cols[0] if num_cols else 'value'}?', "
            f"'How many unique {cat_cols[0] if cat_cols else 'categories'} are there?', "
            f"or 'Show me rows where [column] > [value]'."
        )
        return {"answer": answer, "code": None, "result": None}
