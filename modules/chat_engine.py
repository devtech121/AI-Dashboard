"""
Chat Engine Module
AST-validated sandboxed pandas execution for chat Q&A.

Security: strict AST allowlist, blocked I/O methods, and a hard timeout via a separate process.
"""

import ast
import logging
import multiprocessing as mp
import pandas as pd
import numpy as np
import difflib
from utils.config import AppConfig

logger = logging.getLogger("ai_dashboard.chat_engine")

# Hard timeout for sandbox execution (seconds)
EXEC_TIMEOUT_SECONDS = 8

# -- AST Whitelist ------------------------------------------------------------
ALLOWED_AST_NODES = {
    ast.Assign, ast.AugAssign, ast.Expr, ast.Return,
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call,
    ast.IfExp, ast.Attribute, ast.Subscript, ast.Index, ast.Slice,
    ast.Name, ast.Constant,
    ast.keyword,
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

# Python 3.14 removed ast.Num/ast.Str/ast.NameConstant; only add if present.
for _name in ("Num", "Str", "NameConstant"):
    _node = getattr(ast, _name, None)
    if _node is not None:
        ALLOWED_AST_NODES.add(_node)

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
# Any Name node whose id is not in this set is rejected.
ALLOWED_ROOT_NAMES = {
    "df", "pd", "np", "result", "True", "False", "None",
    "len", "range", "list", "dict", "str", "int", "float", "bool",
    "tuple", "set", "zip", "enumerate", "map", "filter",
    "sorted", "reversed", "min", "max", "sum", "abs", "round",
    "isinstance", "type",
}

# Pandas/numpy METHOD names that perform filesystem I/O
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


def _run_sandboxed_exec(code: str, df: pd.DataFrame, queue: mp.Queue):
    """Execute code in a restricted globals dict and return result via queue."""
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
        exec(compile(code, "<chat>", "exec"), safe_globals)
        queue.put({"success": True, "result": safe_globals.get("result"), "error": None})
    except Exception as e:
        queue.put({"success": False, "result": None, "error": str(e)})


class ChatEngine:
    """Chat-with-data: LLM generates pandas code → AST-validated → timed sandbox."""

    def __init__(self, model: str = None):
        self.model = model or AppConfig.DEFAULT_MODEL

    def ask(self, question: str, df: pd.DataFrame, profile: dict) -> dict:
        rule = self._rule_based_order_lookup(question, df, profile)
        if rule:
            return rule
        col_ctx = self._build_column_context(profile)
        candidates = self._map_question_to_columns(question, profile)
        if self._needs_clarification(question, candidates, profile):
            options = ", ".join([f"`{c}`" for c in candidates[:4]])
            return {
                "answer": f"I found multiple possible columns for your question. Which should I use? {options}",
                "code": None,
                "result": None,
                "evidence": None,
            }
        hints = self._parse_hints_from_text(question, df, profile, candidates)
        spec = self._generate_query_spec(question, df, profile, col_ctx, candidates, hints)
        if spec:
            code = self._generate_code_from_spec(question, df, profile, col_ctx, spec)
        else:
            code = self._generate_code(question, df, profile)
        if code:
            result = self._safe_execute(code, df)
            if result["success"]:
                evidence = self._build_evidence(result["result"])
                if spec and not self._validate_result_against_spec(result["result"], spec):
                    fixed = self._repair_code(question, df, profile, col_ctx, spec, code)
                    if fixed:
                        result2 = self._safe_execute(fixed, df)
                        if result2["success"]:
                            return {
                                "answer": self._format_answer(question, result2["result"], fixed),
                                "code": fixed,
                                "result": result2["result"],
                                "evidence": self._build_evidence(result2["result"]),
                            }
                return {
                    "answer": self._format_answer(question, result["result"], code),
                    "code": code,
                    "result": result["result"],
                    "evidence": evidence,
                }
            logger.warning(f"Exec failed: {result['error']}")
            fallback = self._rule_based_answer(question, df, profile)
            if fallback:
                return fallback
        return self._general_fallback(question, df, profile)

    # ── Code Generation ────────────────────────────────────────

    def _build_column_context(self, profile: dict) -> str:
        lines = []
        for col in profile.get("columns", [])[:20]:
            name = col.get("name")
            ctype = col.get("type")
            dtype = col.get("dtype")
            line = f"- {name} ({ctype}, {dtype})"
            if ctype == "categorical":
                top = list((col.get("top_values") or {}).keys())[:3]
                if top:
                    line += f" values={top}"
            lines.append(line)
        return "\n".join(lines)

    def _needs_clarification(self, question: str, candidates: list, profile: dict) -> bool:
        if not question or len(candidates) < 2:
            return False
        if self._question_mentions_column(question, profile):
            return False
        q = question.lower()
        if "order" in q and any("order" in c.lower() and "id" in c.lower() for c in candidates):
            return False
        return True

    def _question_mentions_column(self, question: str, profile: dict) -> bool:
        q = (question or "").lower()
        for col in profile.get("columns", []):
            name = str(col.get("name", ""))
            if name and name.lower() in q:
                return True
        return False

    def _build_evidence(self, result):
        try:
            if isinstance(result, pd.DataFrame):
                return result.head(5)
            if isinstance(result, pd.Series):
                return result.head(5).to_frame().reset_index()
        except Exception:
            return None
        return None

    def _rule_based_order_lookup(self, question: str, df: pd.DataFrame, profile: dict) -> dict | None:
        col, value = self._detect_order_id_query(question, profile)
        if not col or value is None:
            return None
        try:
            series = df[col]
            if series.dtype.kind in {"i", "u", "f"}:
                try:
                    val = int(value)
                except Exception:
                    val = float(value)
            else:
                val = str(value)
            result_df = df[df[col] == val]
            code = f"result = df[df['{col}'] == {repr(val)}]"
            if len(result_df) == 0:
                return {
                    "answer": f"No records found for `{col}` = {val}.",
                    "code": code,
                    "result": result_df,
                    "evidence": None,
                }
            return {
                "answer": f"Found {len(result_df):,} record(s) for `{col}` = {val}.",
                "code": code,
                "result": result_df,
                "evidence": result_df.head(5),
            }
        except Exception as e:
            logger.warning(f"Order lookup failed: {e}")
            return None

    def _detect_order_id_query(self, question: str, profile: dict) -> tuple:
        if not question:
            return None, None
        q = question.lower()
        cols = [c.get("name") for c in profile.get("columns", []) if c.get("name")]
        col_map = {c.lower(): c for c in cols}
        candidates = []
        for key in ["order_id", "orderid", "order-id", "order id"]:
            if key in col_map:
                candidates.append(col_map[key])
        if not candidates:
            for c in cols:
                if "order" in c.lower() and "id" in c.lower():
                    candidates.append(c)
        if not candidates:
            return None, None

        import re
        m = re.search(r"\border\s*[_-]?\s*id\b\s*[:=]?\s*([0-9A-Za-z_-]+)", q)
        if m:
            return candidates[0], m.group(1)
        m = re.search(r"\border\b\s*([0-9A-Za-z_-]+)\b", q)
        if m:
            return candidates[0], m.group(1)
        return None, None

    def _map_question_to_columns(self, question: str, profile: dict) -> list:
        if not question:
            return []
        q = question.lower()
        cols = [c.get("name") for c in profile.get("columns", []) if c.get("name")]
        col_lowers = {c.lower(): c for c in cols}
        hits = []

        for c in cols:
            cl = c.lower()
            if cl in q:
                hits.append(c)
                continue
            parts = cl.replace("-", " ").replace("_", " ").split()
            if any(p and p in q for p in parts):
                hits.append(c)

        synonyms = {
            "revenue": ["sales", "turnover"],
            "sales": ["revenue", "turnover"],
            "amount": ["total", "sum"],
            "price": ["cost", "amount", "value"],
            "profit": ["margin", "earnings"],
            "quantity": ["qty", "volume", "count"],
            "date": ["time", "day", "month", "year"],
            "customer": ["client", "buyer", "user"],
            "product": ["item", "sku"],
            "category": ["type", "segment", "group"],
            "region": ["state", "country", "area", "zone"],
            "department": ["team", "function", "division"],
            "age": ["years", "dob"],
            "gender": ["sex"],
        }
        for col in cols:
            cl = col.lower()
            for key, syns in synonyms.items():
                if key in cl and any(s in q for s in syns + [key]):
                    hits.append(col)

        q_tokens = [t for t in q.replace("?", " ").replace(",", " ").split() if len(t) >= 3]
        for t in q_tokens[:6]:
            close = difflib.get_close_matches(t, list(col_lowers.keys()), n=1, cutoff=0.8)
            if close:
                hits.append(col_lowers[close[0]])

        seen = set()
        out = []
        for h in hits:
            if h not in seen:
                seen.add(h)
                out.append(h)
        return out[:6]

    def _generate_query_spec(
        self,
        question: str,
        df: pd.DataFrame,
        profile: dict,
        col_ctx: str,
        candidates: list,
        hints: dict,
    ) -> dict | None:
        try:
            from modules.ai_engine import AIEngine
            ai = AIEngine(model=self.model)
            cols = [c.get("name") for c in profile.get("columns", []) if c.get("name")]
            prompt = f"""Return ONLY valid JSON (no markdown). Create a query spec for a pandas DataFrame.

DataFrame: df ({df.shape[0]} rows x {df.shape[1]} cols)
Columns:
{col_ctx}

User question: {question}
Candidate columns (from semantic match): {candidates}
Hints (parsed from text): {hints}

Rules:
1. Use ONLY columns from this list: {cols}
2. If unsure, choose the most likely column from candidates.
3. Keep it minimal and consistent.
4. If the question has multiple parts, include all required filters/grouping in the spec.

Schema:
{{
  "operation": "aggregate|list|filter|describe|other",
  "target_columns": ["col1","col2"],
  "groupby": ["col_or_empty"],
  "filters": [{{"column":"col","op":"==|!=|>|>=|<|<=|contains|in","value":"..."}}],
  "sort": {{"column":"col_or_empty","order":"asc|desc"}},
  "limit": 0,
  "output": "scalar|series|table",
  "assumptions": "short text"
}}
"""
            raw = ai.call(prompt=prompt, max_tokens=300, temperature=0.05)
            spec = self._extract_json(raw)
            if not isinstance(spec, dict):
                return None
            return self._sanitize_spec(spec, cols, candidates, hints)
        except Exception as e:
            logger.warning(f"Spec generation failed: {e}")
            return None

    def _extract_json(self, text: str) -> dict | None:
        if not text:
            return None
        import json, re
        try:
            return json.loads(text)
        except Exception:
            pass
        for pat in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            m = re.search(pat, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                return None
        return None

    def _sanitize_spec(self, spec: dict, valid_cols: list, candidates: list, hints: dict) -> dict | None:
        if not isinstance(spec, dict):
            return None
        def _clean_cols(cols):
            out = []
            for c in cols or []:
                if c in valid_cols and c not in out:
                    out.append(c)
            return out
        spec["target_columns"] = _clean_cols(spec.get("target_columns", []))
        spec["groupby"] = _clean_cols(spec.get("groupby", []))
        filt = []
        for f in spec.get("filters", []) or []:
            col = f.get("column")
            if col in valid_cols:
                filt.append({"column": col, "op": f.get("op", "=="), "value": f.get("value", "")})
        spec["filters"] = filt
        sort = spec.get("sort", {}) or {}
        if sort.get("column") not in valid_cols:
            sort = {"column": "", "order": "asc"}
        spec["sort"] = {"column": sort.get("column", ""), "order": sort.get("order", "asc")}
        if hints:
            if not spec.get("filters") and hints.get("filters"):
                spec["filters"] = hints["filters"]
            if spec.get("limit", 0) in (0, None) and hints.get("limit"):
                spec["limit"] = hints["limit"]
            if (not spec.get("sort") or not spec["sort"].get("column")) and hints.get("sort"):
                spec["sort"] = hints["sort"]
        if not spec["target_columns"] and candidates:
            spec["target_columns"] = [c for c in candidates if c in valid_cols][:2]
        if not spec["target_columns"]:
            return None
        return spec

    def _parse_hints_from_text(self, question: str, df: pd.DataFrame, profile: dict, candidates: list) -> dict:
        if not question:
            return {}
        q = question.lower()
        hints = {"filters": [], "sort": {"column": "", "order": "asc"}, "limit": 0}
        cols = [c.get("name") for c in profile.get("columns", []) if c.get("name")]
        col_types = {c.get("name"): c.get("type") for c in profile.get("columns", [])}

        import re
        m = re.search(r"\btop\s+(\d+)", q)
        if m:
            hints["limit"] = int(m.group(1))
            hints["sort"]["order"] = "desc"
        m = re.search(r"\b(bottom|lowest)\s+(\d+)", q)
        if m:
            hints["limit"] = int(m.group(2))
            hints["sort"]["order"] = "asc"

        # Numeric comparisons
        comp_map = {
            r"\bgreater than\s+(-?\d+\.?\d*)": ">",
            r"\bmore than\s+(-?\d+\.?\d*)": ">",
            r"\bover\s+(-?\d+\.?\d*)": ">",
            r"\bless than\s+(-?\d+\.?\d*)": "<",
            r"\bunder\s+(-?\d+\.?\d*)": "<",
            r"\bbetween\s+(-?\d+\.?\d*)\s+and\s+(-?\d+\.?\d*)": "between",
        }
        target_col = None
        for c in candidates:
            if col_types.get(c) == "numerical":
                target_col = c
                break
        if not target_col:
            for c in cols:
                if col_types.get(c) == "numerical" and c.lower() in q:
                    target_col = c
                    break
        for pat, op in comp_map.items():
            m = re.search(pat, q)
            if m and target_col:
                if op == "between":
                    hints["filters"].append({"column": target_col, "op": ">=", "value": m.group(1)})
                    hints["filters"].append({"column": target_col, "op": "<=", "value": m.group(2)})
                else:
                    hints["filters"].append({"column": target_col, "op": op, "value": m.group(1)})
                break

        # Last N days for datetime columns
        m = re.search(r"\blast\s+(\d+)\s+days\b", q)
        dt_cols = profile.get("datetime_cols", [])
        if m and dt_cols:
            try:
                days = int(m.group(1))
                dt_col = dt_cols[0]
                series = pd.to_datetime(df[dt_col], errors="coerce").dropna()
                if len(series) > 0:
                    max_dt = series.max()
                    cutoff = (max_dt - pd.Timedelta(days=days)).date().isoformat()
                    hints["filters"].append({"column": dt_col, "op": ">=", "value": cutoff})
            except Exception:
                pass

        # Sort hint based on phrasing "by <col>"
        m = re.search(r"\bby\s+([a-zA-Z0-9_ -]+)$", q)
        if m:
            name = m.group(1).strip().replace(" ", "_")
            for c in cols:
                if c.lower() == name.lower():
                    hints["sort"]["column"] = c
                    break

        return hints

    def _generate_code_from_spec(
        self,
        question: str,
        df: pd.DataFrame,
        profile: dict,
        col_ctx: str,
        spec: dict,
    ) -> str | None:
        try:
            from modules.ai_engine import AIEngine
            ai = AIEngine(model=self.model)
            prompt = f"""Generate ONLY executable pandas code. No explanation. No markdown.

DataFrame: df ({df.shape[0]} rows x {df.shape[1]} cols)
Columns:
{col_ctx}

User question: {question}
Query spec (must follow): {spec}

Rules:
1. Store result in variable named `result`
2. Only use: df, pd, np, len, sum, min, max, sorted, round, list, dict, str, int, float
3. No imports, no open(), no read_csv(), no to_csv(), no file I/O of any kind
4. No for/while loops, no class/def, no exec/eval
5. result = DataFrame | Series | scalar | string
6. Always .reset_index() after groupby
7. Honor filters, groupby, sort, and limit from the spec

Code:"""
            response = ai.call(prompt=prompt, max_tokens=320, temperature=0.05)
            return self._extract_code(response) if response else None
        except Exception as e:
            logger.error(f"Spec code generation: {e}")
            return None

    def _validate_result_against_spec(self, result, spec: dict) -> bool:
        if not spec:
            return True
        expected = spec.get("output", "table")
        if expected == "scalar":
            return isinstance(result, (int, float, np.integer, np.floating, str))
        if expected == "series":
            return isinstance(result, pd.Series)
        if expected == "table":
            return isinstance(result, (pd.DataFrame, pd.Series))
        return True

    def _repair_code(
        self,
        question: str,
        df: pd.DataFrame,
        profile: dict,
        col_ctx: str,
        spec: dict,
        bad_code: str,
    ) -> str | None:
        try:
            from modules.ai_engine import AIEngine
            ai = AIEngine(model=self.model)
            prompt = f"""Fix the pandas code to satisfy the query spec. Output ONLY corrected code.

User question: {question}
Query spec: {spec}
Columns:
{col_ctx}

Bad code:
{bad_code}

Rules:
1. Store result in variable named `result`
2. Only use: df, pd, np, len, sum, min, max, sorted, round, list, dict, str, int, float
3. No imports, no open(), no read_csv(), no to_csv(), no file I/O of any kind
4. No for/while loops, no class/def, no exec/eval
5. result = DataFrame | Series | scalar | string
6. Always .reset_index() after groupby
"""
            response = ai.call(prompt=prompt, max_tokens=260, temperature=0.05)
            return self._extract_code(response) if response else None
        except Exception as e:
            logger.error(f"Repair code failed: {e}")
            return None

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
          - multiprocessing-based hard kill after EXEC_TIMEOUT_SECONDS
        """
        is_safe, reason = _validate_ast(code)
        if not is_safe:
            logger.warning(f"AST rejected: {reason}")
            return {"success": False, "result": None, "error": f"Code rejected: {reason}"}

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_run_sandboxed_exec, args=(code, df, queue), daemon=True)
        proc.start()
        proc.join(timeout=EXEC_TIMEOUT_SECONDS)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)
            logger.warning(f"Chat code timed out after {EXEC_TIMEOUT_SECONDS}s")
            return {
                "success": False,
                "result": None,
                "error": f"Query took too long (>{EXEC_TIMEOUT_SECONDS}s). Try a simpler question.",
            }

        try:
            outcome = queue.get(timeout=1)
        except Exception:
            outcome = {"success": False, "result": None, "error": "No result returned from sandbox"}

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


