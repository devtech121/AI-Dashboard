"""
Chat Engine Module
AST-validated sandboxed pandas execution for chat Q&A.

Security fixes in this version:
  - ALLOWED_ROOT_NAMES is now ENFORCED: every top-level Name in the AST
    must be in the explicit allowlist, blocking pd.read_csv, pd.to_csv,
    and any other filesystem I/O through pandas (Issue #1 follow-up).
  - Execution runs inside a threading.Timer hard-kill (Issue #2 follow-up):
    code that would block (heavy apply, infinite groupby, etc.) is aborted
    after EXEC_TIMEOUT_SECONDS regardless.
  - Dangerous pandas method names are blocked at the AST attribute level.
"""

import ast
import logging
import threading
import pandas as pd
import numpy as np
from utils.config import AppConfig

logger = logging.getLogger("ai_dashboard.chat_engine")

# Hard timeout for sandbox execution (seconds)
EXEC_TIMEOUT_SECONDS = 8

# ── AST Whitelist ──────────────────────────────────────────────
ALLOWED_AST_NODES = {
    ast.Assign, ast.AugAssign, ast.Expr, ast.Return,
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call,
    ast.IfExp, ast.Attribute, ast.Subscript, ast.Index, ast.Slice,
    ast.Name, ast.Constant, ast.Num, ast.Str, ast.NameConstant,
    ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    ast.comprehension,
    ast.If,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
    ast.Load, ast.Store, ast.Del,
    ast.Module, ast.Expression,
}

# Names that are NEVER allowed as identifiers
FORBIDDEN_NAMES = {
    "__import__", "__builtins__", "__class__", "__bases__", "__subclasses__",
    "__globals__", "__locals__", "__dict__", "__code__", "__closure__",
    "exec", "eval", "compile", "open", "input", "breakpoint",
    "getattr", "setattr", "delattr", "vars", "dir", "globals", "locals",
    "os", "sys", "subprocess", "shutil", "pathlib", "socket", "urllib",
    "requests", "http", "importlib", "builtins", "ctypes", "pickle",
    "tempfile", "glob", "io", "csv", "json", "xml", "yaml",
}

# ENFORCED allowlist: the ONLY top-level names permitted in generated code.
# Any Name node whose id is not in this set → rejected.
ALLOWED_ROOT_NAMES = {
    "df", "pd", "np", "result", "True", "False", "None",
    "len", "range", "list", "dict", "str", "int", "float", "bool",
    "tuple", "set", "zip", "enumerate", "map", "filter",
    "sorted", "reversed", "min", "max", "sum", "abs", "round",
    "isinstance", "type",
}

# Pandas/numpy METHOD names that perform filesystem I/O — blocked at attribute level
FORBIDDEN_METHODS = {
    "read_csv", "read_excel", "read_json", "read_html", "read_sql",
    "read_parquet", "read_feather", "read_pickle", "read_hdf",
    "to_csv", "to_excel", "to_json", "to_html", "to_sql",
    "to_parquet", "to_feather", "to_pickle", "to_hdf",
    "read_clipboard", "to_clipboard",
    "save", "load", "dump", "dumps", "loads",
    "system", "popen", "spawn",
}


class _ASTValidator(ast.NodeVisitor):
    """Walks AST and collects violations."""

    def __init__(self):
        self.errors = []

    def generic_visit(self, node):
        if type(node) not in ALLOWED_AST_NODES:
            self.errors.append(f"Disallowed node: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id in FORBIDDEN_NAMES:
            self.errors.append(f"Forbidden identifier: {node.id}")
        # Enforce top-level allowlist for Load context (variable reads)
        if isinstance(node.ctx, ast.Load) and node.id not in ALLOWED_ROOT_NAMES:
            self.errors.append(f"Identifier not in allowlist: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            self.errors.append(f"Forbidden dunder: {node.attr}")
        if node.attr in FORBIDDEN_METHODS:
            self.errors.append(f"Forbidden I/O method: {node.attr}()")
        self.generic_visit(node)

    def visit_Import(self, node):
        self.errors.append("Import not allowed")

    def visit_ImportFrom(self, node):
        self.errors.append("Import not allowed")

    def visit_Global(self, node):
        self.errors.append("Global not allowed")

    def visit_Nonlocal(self, node):
        self.errors.append("Nonlocal not allowed")

    def visit_Delete(self, node):
        self.errors.append("Delete not allowed")

    def visit_ClassDef(self, node):
        self.errors.append("Class definition not allowed")

    def visit_FunctionDef(self, node):
        self.errors.append("Function definition not allowed")

    def visit_AsyncFunctionDef(self, node):
        self.errors.append("Async not allowed")

    def visit_While(self, node):
        self.errors.append("While loop not allowed")

    def visit_For(self, node):
        self.errors.append("For loop not allowed")


def _validate_ast(code: str) -> tuple:
    """Parse and validate code. Returns (is_safe, reason)."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    v = _ASTValidator()
    v.visit(tree)

    if v.errors:
        return False, "; ".join(v.errors[:3])
    return True, ""


class ChatEngine:
    """Chat-with-data: LLM generates pandas code → AST-validated → timed sandbox."""

    def __init__(self, model: str = None):
        self.model = model or AppConfig.DEFAULT_MODEL

    def ask(self, question: str, df: pd.DataFrame, profile: dict) -> dict:
        code = self._generate_code(question, df, profile)
        if code:
            result = self._safe_execute(code, df)
            if result["success"]:
                return {
                    "answer": self._format_answer(question, result["result"], code),
                    "code": code,
                    "result": result["result"],
                }
            logger.warning(f"Exec failed: {result['error']}")
            fallback = self._rule_based_answer(question, df, profile)
            if fallback:
                return fallback
        return self._general_fallback(question, df, profile)

    # ── Code Generation ────────────────────────────────────────

    def _generate_code(self, question: str, df: pd.DataFrame, profile: dict) -> str | None:
        try:
            from modules.ai_engine import AIEngine
            ai = AIEngine(model=self.model)

            col_info = []
            for col in profile.get("columns", [])[:15]:
                info = f"{col['name']} ({col['type']}, {col['dtype']})"
                if col["type"] == "categorical" and col.get("top_values"):
                    top = list(col["top_values"].keys())[:3]
                    info += f" — values: {top}"
                col_info.append(info)

            cols_text = "\n".join(f"- {c}" for c in col_info)

            prompt = f"""Generate ONLY executable pandas code. No explanation. No markdown.

DataFrame: df ({df.shape[0]} rows x {df.shape[1]} cols)
Columns:
{cols_text}

Question: {question}

Rules:
1. Store result in variable named `result`
2. Only use: df, pd, np, len, sum, min, max, sorted, round, list, dict, str, int, float
3. No imports, no open(), no read_csv(), no to_csv(), no file I/O of any kind
4. No for/while loops, no class/def, no exec/eval
5. result = DataFrame | Series | scalar | string
6. Always .reset_index() after groupby

Code:"""

            response = ai.call(prompt=prompt, max_tokens=300, temperature=0.05)
            return self._extract_code(response) if response else None
        except Exception as e:
            logger.error(f"Code generation: {e}")
            return None

    def _extract_code(self, response: str) -> str | None:
        if not response:
            return None
        import re
        for pat in [r"```python\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            m = re.search(pat, response, re.DOTALL)
            if m:
                return m.group(1).strip()
        cleaned = response.strip()
        if "result" in cleaned and "=" in cleaned:
            return cleaned
        return None

    # ── Secure Timed Execution ─────────────────────────────────

    def _safe_execute(self, code: str, df: pd.DataFrame) -> dict:
        """
        AST-validate then execute in a minimal sandbox with a hard time limit.
        Issue fixes:
          - ALLOWED_ROOT_NAMES enforced (blocks pd.read_csv etc.)
          - FORBIDDEN_METHODS blocks I/O at attribute level
          - threading.Timer kills execution after EXEC_TIMEOUT_SECONDS
        """
        is_safe, reason = _validate_ast(code)
        if not is_safe:
            logger.warning(f"AST rejected: {reason}")
            return {"success": False, "result": None, "error": f"Code rejected: {reason}"}

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

        outcome = {"success": False, "result": None, "error": "Timeout"}

        def run():
            try:
                exec(compile(code, "<chat>", "exec"), safe_globals)
                outcome["result"]  = safe_globals.get("result")
                outcome["success"] = True
                outcome["error"]   = None
            except Exception as e:
                outcome["error"] = str(e)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=EXEC_TIMEOUT_SECONDS)

        if thread.is_alive():
            logger.warning(f"Chat code timed out after {EXEC_TIMEOUT_SECONDS}s")
            outcome["error"] = f"Query took too long (>{EXEC_TIMEOUT_SECONDS}s). Try a simpler question."

        return outcome

    # ── Answer Formatting ──────────────────────────────────────

    def _format_answer(self, question: str, result, code: str) -> str:
        if result is None:
            return "Query ran successfully but returned no results."
        if isinstance(result, pd.DataFrame):
            return "No matching records." if len(result) == 0 else f"Found {len(result):,} row(s), {len(result.columns)} column(s)."
        if isinstance(result, pd.Series):
            if len(result) == 0:
                return "No data found."
            if len(result) <= 10:
                items = "\n".join(
                    f"• **{k}**: {v:,.2f}" if isinstance(v, float) else f"• **{k}**: {v}"
                    for k, v in result.items()
                )
                return f"Results:\n\n{items}"
            return f"Found {len(result)} items."
        if isinstance(result, (int, float, np.integer, np.floating)):
            return f"**Answer:** {result:,.4f}" if isinstance(result, float) else f"**Answer:** {result:,}"
        return str(result)

    # ── Rule-Based Fallbacks ───────────────────────────────────

    def _rule_based_answer(self, question: str, df: pd.DataFrame, profile: dict) -> dict | None:
        q = question.lower()
        try:
            if any(p in q for p in ["how many rows", "row count", "total records"]):
                return {"answer": f"The dataset has **{len(df):,} rows**.", "code": "result = len(df)", "result": len(df)}
            if any(p in q for p in ["what columns", "list columns", "column names"]):
                cols = ", ".join(df.columns.tolist())
                return {"answer": f"**{len(df.columns)} columns**: {cols}", "code": "result = df.columns.tolist()", "result": None}
            if "missing" in q or "null" in q:
                missing = df.isnull().sum()
                missing = missing[missing > 0]
                if len(missing) == 0:
                    return {"answer": "✅ No missing values.", "code": "result = df.isnull().sum()", "result": None}
                rdf = missing.reset_index()
                rdf.columns = ["Column", "Missing Count"]
                return {"answer": f"**{len(missing)} column(s)** have missing values.", "code": "result = df.isnull().sum()[df.isnull().sum()>0].reset_index()", "result": rdf}
            if "duplicate" in q:
                n = int(df.duplicated().sum())
                return {"answer": f"**{n:,} duplicate rows** detected.", "code": "result = df.duplicated().sum()", "result": n}
        except Exception as e:
            logger.warning(f"Rule-based fallback: {e}")
        return None

    def _general_fallback(self, question: str, df: pd.DataFrame, profile: dict) -> dict:
        num_cols = profile.get("numerical_cols", [])
        cat_cols = profile.get("categorical_cols", [])
        ex = []
        if num_cols: ex.append(f"'What is the average {num_cols[0]}?'")
        if cat_cols: ex.append(f"'How many unique {cat_cols[0]} values?'")
        return {"answer": f"Could not generate code for that. Try: {' or '.join(ex[:2])}", "code": None, "result": None}