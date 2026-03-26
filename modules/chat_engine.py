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
import re
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
        rule = self._rule_based_category_count(question, df, profile)
        if rule:
            return rule
        rule = self._rule_based_return_count(question, df, profile)
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
                auto_explain = self._build_auto_explain(question, result["result"], spec, df, profile)
                corrected = self._semantic_correction(question, result["result"], df, profile, spec)
                if corrected:
                    return corrected
                if spec and not self._validate_result_against_spec(result["result"], spec):
                    fixed = self._repair_code(question, df, profile, col_ctx, spec, code)
                    if fixed:
                        result2 = self._safe_execute(fixed, df)
                        if result2["success"]:
                            auto_explain2 = self._build_auto_explain(question, result2["result"], spec, df, profile)
                            return {
                                "answer": self._format_answer(question, result2["result"], fixed) + (f"\n\n{auto_explain2}" if auto_explain2 else ""),
                                "code": fixed,
                                "result": result2["result"],
                                "evidence": self._build_evidence(result2["result"]),
                                "confidence": 0.6,
                                "source": "llm-repaired",
                                "confidence_reason": "LLM result repaired to match spec",
                            }
                return {
                    "answer": self._format_answer(question, result["result"], code) + (f"\n\n{auto_explain}" if auto_explain else ""),
                    "code": code,
                    "result": result["result"],
                    "evidence": evidence,
                    "confidence": 0.7 if spec else 0.6,
                    "source": "llm",
                    "confidence_reason": "LLM-generated code executed successfully",
                }
            logger.warning(f"Exec failed: {result['error']}")
            fallback = self._rule_based_answer(question, df, profile)
            if fallback:
                fallback["confidence"] = 0.5
                fallback["source"] = "rule-fallback"
                fallback["confidence_reason"] = "Rule-based fallback"
                return fallback
        fb = self._general_fallback(question, df, profile)
        fb["confidence"] = 0.2
        fb["source"] = "fallback"
        fb["confidence_reason"] = "Could not infer a safe computation"
        return fb

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

    def _is_numeric_col(self, col: str, profile: dict) -> bool:
        num_cols = set(profile.get("numerical_cols", []))
        return col in num_cols

    def _build_auto_explain(self, question: str, result, spec: dict, df: pd.DataFrame, profile: dict) -> str | None:
        try:
            if isinstance(result, pd.DataFrame):
                cols_lower = [c.lower() for c in result.columns]
                if "count" in cols_lower:
                    count_col = result.columns[cols_lower.index("count")]
                    total = result[count_col].sum()
                    if total and len(result) >= 1:
                        top_n = min(5, len(result))
                        top_share = result[count_col].head(top_n).sum() / total * 100
                        return f"Top {top_n} categories account for about {top_share:.1f}% of all records."
        except Exception:
            return None
        return None

    def _semantic_correction(self, question: str, result, df: pd.DataFrame, profile: dict, spec: dict):
        # If the question implies a deterministic count table but result isn't one, override.
        try:
            if self._looks_like_category_count(question):
                if not self._is_count_table(result):
                    rule = self._rule_based_category_count(question, df, profile)
                    if rule:
                        rule["confidence_reason"] = "Rule-based override: count intent detected"
                        return rule
            if spec and spec.get("aggregation") in ("sum", "mean", "min", "max"):
                if not self._result_is_numeric(result):
                    fixed = self._deterministic_aggregate(df, spec)
                    if fixed:
                        fixed["confidence_reason"] = "Deterministic aggregate override for numeric accuracy"
                        return fixed
        except Exception:
            pass
        return None

    def _result_is_numeric(self, result) -> bool:
        if isinstance(result, (int, float, np.integer, np.floating)):
            return True
        if isinstance(result, pd.Series):
            return result.dtype.kind in "if"
        if isinstance(result, pd.DataFrame):
            return len(result.select_dtypes(include=["number"]).columns) > 0
        return False

    def _deterministic_aggregate(self, df: pd.DataFrame, spec: dict) -> dict | None:
        try:
            target_cols = spec.get("target_columns") or []
            groupby = spec.get("groupby") or []
            agg = spec.get("aggregation")
            if not target_cols:
                return None
            col = target_cols[0]
            if groupby:
                gcol = groupby[0]
                agg_df = getattr(df.groupby(gcol)[col], agg)().reset_index()
                agg_df.columns = [gcol, col]
                return {
                    "answer": f"Found {len(agg_df):,} group(s) with {agg} for {col}.",
                    "code": f"result = df.groupby('{gcol}')['{col}'].{agg}().reset_index()",
                    "result": agg_df,
                    "evidence": agg_df.head(5),
                    "confidence": 0.85,
                    "source": "rule",
                    "confidence_reason": "Deterministic aggregate override for numeric accuracy",
                }
            val = getattr(df[col], agg)()
            return {
                "answer": f"**Answer:** {val:,.4f}" if isinstance(val, float) else f"**Answer:** {val:,}",
                "code": f"result = df['{col}'].{agg}()",
                "result": val,
                "evidence": None,
                "confidence": 0.85,
                "source": "rule",
                "confidence_reason": "Deterministic aggregate override for numeric accuracy",
            }
        except Exception:
            return None


    def _looks_like_category_count(self, question: str) -> bool:
        if not question:
            return False
        q = question.lower()
        patterns = [
            r"\bhow many\b.*\bcount\b",
            r"\bhow many\b.*\bdifferent\b",
            r"\bcount\b.*\bby\b",
            r"\bdistribution\b",
            r"\btheir count\b",
        ]
        return any(re.search(p, q) for p in patterns)

    def _is_count_table(self, result) -> bool:
        if isinstance(result, pd.DataFrame):
            cols_lower = [c.lower() for c in result.columns]
            return any(c in cols_lower for c in ["count", "counts", "n", "num"])
        return False

    def _parse_numeric_value(self, raw: str) -> float | None:
        if raw is None:
            return None
        s = str(raw).strip().lower().replace(",", "")
        s = re.sub(r"[$\u20b9\u20ac\u00a3]", "", s)
        mult = 1.0
        if s.endswith("k"):
            mult = 1_000.0
            s = s[:-1]
        elif s.endswith("m"):
            mult = 1_000_000.0
            s = s[:-1]
        elif s.endswith("b"):
            mult = 1_000_000_000.0
            s = s[:-1]
        try:
            return float(s) * mult
        except Exception:
            return None

    def _rank_columns(self, question: str, profile: dict) -> list:
        if not question:
            return []
        q = question.lower()
        cols = [c.get("name") for c in profile.get("columns", []) if c.get("name")]
        synonyms = {
            "revenue": ["sales", "turnover"],
            "sales": ["revenue", "turnover"],
            "amount": ["total", "sum", "value"],
            "price": ["cost", "amount", "value"],
            "profit": ["margin", "earnings"],
            "quantity": ["qty", "volume", "count"],
            "brand": ["make", "manufacturer"],
            "make": ["brand", "manufacturer"],
            "model": ["variant", "trim"],
            "date": ["time", "day", "month", "year", "quarter"],
            "customer": ["client", "buyer", "user"],
            "product": ["item", "sku"],
            "category": ["type", "segment", "group", "class"],
            "region": ["state", "country", "area", "zone"],
            "payment": ["method", "payment_method"],
        }
        q_tokens = [t for t in re.split(r"\W+", q) if len(t) >= 3]
        scored = []
        for c in cols:
            cl = c.lower()
            score = 0.0
            if cl in q:
                score += 3.0
            parts = cl.replace("-", " ").replace("_", " ").split()
            score += sum(1.0 for p in parts if p in q_tokens)
            for key, syns in synonyms.items():
                if key in cl and any(s in q for s in syns + [key]):
                    score += 2.0
            for t in q_tokens[:6]:
                score += difflib.SequenceMatcher(None, t, cl).ratio() * 0.5
            scored.append((score, c))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for s, c in scored if s > 0.2]

    def _column_type_map(self, profile: dict) -> dict:
        return {c.get("name"): c.get("type") for c in profile.get("columns", []) if c.get("name")}

    def _rule_based_category_count(self, question: str, df: pd.DataFrame, profile: dict) -> dict | None:
        if not question:
            return None
        q = question.lower()
        import re
        count_patterns = [
            r"\bhow many\b.*\bcount\b",
            r"\bhow many\b.*\bdifferent\b",
            r"\bcount\b.*\bby\b",
            r"\bdistribution\b",
            r"\btheir count\b",
        ]
        if not any(re.search(p, q) for p in count_patterns):
            return None

        cat_cols = profile.get("categorical_cols", [])
        type_map = self._column_type_map(profile)
        candidates = self._map_question_to_columns(question, profile)

        # Prefer explicit categorical candidates
        for c in candidates:
            if type_map.get(c) == "categorical":
                return self._build_count_response(df, c)

        # Synonym-based match for common category terms
        synonyms = {
            "brand": ["brand", "make", "manufacturer"],
            "model": ["model", "variant"],
            "category": ["category", "type", "segment", "class"],
            "region": ["region", "state", "country", "area", "zone"],
            "payment": ["payment", "payment_method", "method"],
            "product": ["product", "item", "sku"],
        }
        for col in cat_cols:
            cl = col.lower()
            for _, syns in synonyms.items():
                if any(s in q for s in syns) and any(s in cl for s in syns):
                    return self._build_count_response(df, col)

        # Fallback: first categorical column
        if cat_cols:
            return self._build_count_response(df, cat_cols[0])
        return None

    def _rule_based_return_count(self, question: str, df: pd.DataFrame, profile: dict) -> dict | None:
        if not question:
            return None
        q = question.lower()
        if not any(k in q for k in ["return", "returned", "returns"]):
            return None
        cols = [c.get("name") for c in profile.get("columns", []) if c.get("name")]
        # Prefer explicit 'returned' or 'return' boolean/binary columns
        candidates = [c for c in cols if "return" in c.lower()]
        if not candidates:
            return None
        col = candidates[0]
        try:
            series = df[col]
            if series.dtype.kind in {"i", "u", "f"}:
                count_val = int((series == 1).sum())
                code = f"result = (df['{col}'] == 1).sum()"
            else:
                count_val = int((series == True).sum())  # noqa: E712
                code = f"result = (df['{col}'] == True).sum()"
            return {
                "answer": f"**Returned orders:** {count_val:,}",
                "code": code,
                "result": count_val,
                "evidence": None,
                "confidence": 0.95,
                "source": "rule",
                "confidence_reason": "Rule-based deterministic answer",
            }
        except Exception:
            return None

    def _build_count_response(self, df: pd.DataFrame, col: str) -> dict:
        vc = df[col].value_counts(dropna=False).reset_index()
        vc.columns = [col, "count"]
        return {
            "answer": f"Found {len(vc):,} unique {col} values with counts.\n\nTop 5 categories account for about {(vc['count'].head(5).sum() / vc['count'].sum() * 100):.1f}% of all records.",
            "code": f"result = df['{col}'].value_counts(dropna=False).reset_index().rename(columns={{'index':'{col}', '{col}':'count'}})",
            "result": vc,
            "evidence": vc.head(5),
            "confidence": 0.95,
            "source": "rule",
            "confidence_reason": "Rule-based deterministic answer",
        }

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
                    "confidence": 0.95,
                    "source": "rule",
                    "confidence_reason": "Rule-based deterministic answer",
                }
            return {
                "answer": f"Found {len(result_df):,} record(s) for `{col}` = {val}.",
                "code": code,
                "result": result_df,
                "evidence": result_df.head(5),
                "confidence": 0.95,
                "source": "rule",
                "confidence_reason": "Rule-based deterministic answer",
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
        return self._rank_columns(question, profile)[:6]

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
  "aggregation": "count|sum|mean|max|min|none",
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
            return self._sanitize_spec(spec, cols, candidates, hints, profile)
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

    def _sanitize_spec(self, spec: dict, valid_cols: list, candidates: list, hints: dict, profile: dict) -> dict | None:
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
        agg = spec.get("aggregation", "none")
        if agg in ("sum", "mean", "min", "max"):
            num_cols = set(profile.get("numerical_cols", []))
            spec["target_columns"] = [c for c in spec["target_columns"] if c in num_cols]
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
            if hints.get("aggregation") and (not spec.get("aggregation") or spec.get("aggregation") == "none"):
                spec["aggregation"] = hints["aggregation"]
        if not spec["target_columns"] and candidates:
            spec["target_columns"] = [c for c in candidates if c in valid_cols][:2]
        if not spec["target_columns"]:
            return None
        return spec

    def _parse_hints_from_text(self, question: str, df: pd.DataFrame, profile: dict, candidates: list) -> dict:
        if not question:
            return {}
        q = question.lower()
        hints = {"filters": [], "sort": {"column": "", "order": "asc"}, "limit": 0, "aggregation": "none"}
        cols = [c.get("name") for c in profile.get("columns", []) if c.get("name")]
        col_types = {c.get("name"): c.get("type") for c in profile.get("columns", [])}

        import re
        if any(w in q for w in ["average", "avg", "mean"]):
            hints["aggregation"] = "mean"
        elif any(w in q for w in ["sum", "total", "overall total"]):
            hints["aggregation"] = "sum"
        elif any(w in q for w in ["max", "maximum", "highest"]):
            hints["aggregation"] = "max"
        elif any(w in q for w in ["min", "minimum", "lowest"]):
            hints["aggregation"] = "min"
        elif any(w in q for w in ["count", "how many", "number of"]):
            hints["aggregation"] = "count"
        m = re.search(r"\btop\s+(\d+)", q)
        if m:
            hints["limit"] = int(m.group(1))
            hints["sort"]["order"] = "desc"
        m = re.search(r"\b(bottom|lowest)\s+(\d+)", q)
        if m:
            hints["limit"] = int(m.group(2))
            hints["sort"]["order"] = "asc"

        # Numeric comparisons (currency + K/M/B)
        comp_map = {
            r"\bgreater than\s+([\$₹€£]?\s*-?\d[\d,\.]*\s*[kmb]?)": ">",
            r"\bmore than\s+([\$₹€£]?\s*-?\d[\d,\.]*\s*[kmb]?)": ">",
            r"\bover\s+([\$₹€£]?\s*-?\d[\d,\.]*\s*[kmb]?)": ">",
            r"\bless than\s+([\$₹€£]?\s*-?\d[\d,\.]*\s*[kmb]?)": "<",
            r"\bunder\s+([\$₹€£]?\s*-?\d[\d,\.]*\s*[kmb]?)": "<",
            r"\bbetween\s+([\$₹€£]?\s*-?\d[\d,\.]*\s*[kmb]?)\s+and\s+([\$₹€£]?\s*-?\d[\d,\.]*\s*[kmb]?)": "between",
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
                    v1 = self._parse_numeric_value(m.group(1))
                    v2 = self._parse_numeric_value(m.group(2))
                    if v1 is not None and v2 is not None:
                        hints["filters"].append({"column": target_col, "op": ">=", "value": v1})
                        hints["filters"].append({"column": target_col, "op": "<=", "value": v2})
                else:
                    v = self._parse_numeric_value(m.group(1))
                    if v is not None:
                        hints["filters"].append({"column": target_col, "op": op, "value": v})
                break

        # Date ranges for datetime columns
        dt_cols = profile.get("datetime_cols", [])
        if dt_cols:
            dt_col = dt_cols[0]
            series = pd.to_datetime(df[dt_col], errors="coerce").dropna()
            if len(series) > 0:
                max_dt = series.max()
                # last N days
                m = re.search(r"\blast\s+(\d+)\s+days\b", q)
                if m:
                    days = int(m.group(1))
                    cutoff = (max_dt - pd.Timedelta(days=days)).date().isoformat()
                    hints["filters"].append({"column": dt_col, "op": ">=", "value": cutoff})
                # last year
                if "last year" in q:
                    start = pd.Timestamp(max_dt.year - 1, 1, 1).date().isoformat()
                    end = pd.Timestamp(max_dt.year - 1, 12, 31).date().isoformat()
                    hints["filters"].append({"column": dt_col, "op": ">=", "value": start})
                    hints["filters"].append({"column": dt_col, "op": "<=", "value": end})
                # last quarter
                if "last quarter" in q:
                    qtr = (max_dt.month - 1) // 3 + 1
                    last_qtr = qtr - 1 or 4
                    year = max_dt.year if qtr > 1 else max_dt.year - 1
                    start_month = 3 * (last_qtr - 1) + 1
                    start = pd.Timestamp(year, start_month, 1).date().isoformat()
                    end = (pd.Timestamp(year, start_month, 1) + pd.offsets.QuarterEnd()).date().isoformat()
                    hints["filters"].append({"column": dt_col, "op": ">=", "value": start})
                    hints["filters"].append({"column": dt_col, "op": "<=", "value": end})
                # Q1 2024 style
                m = re.search(r"\bq([1-4])\s*(\d{4})\b", q)
                if m:
                    qn = int(m.group(1))
                    year = int(m.group(2))
                    start_month = 3 * (qn - 1) + 1
                    start = pd.Timestamp(year, start_month, 1).date().isoformat()
                    end = (pd.Timestamp(year, start_month, 1) + pd.offsets.QuarterEnd()).date().isoformat()
                    hints["filters"].append({"column": dt_col, "op": ">=", "value": start})
                    hints["filters"].append({"column": dt_col, "op": "<=", "value": end})

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
        agg = spec.get("aggregation", "none")
        if expected == "scalar":
            if agg in ("sum", "mean", "min", "max"):
                return isinstance(result, (int, float, np.integer, np.floating))
            return isinstance(result, (int, float, np.integer, np.floating, str))
        if expected == "series":
            if isinstance(result, pd.Series):
                if agg in ("sum", "mean", "min", "max"):
                    return result.dtype.kind in "if"
                return True
            return False
        if expected == "table":
            if not isinstance(result, (pd.DataFrame, pd.Series)):
                return False
            if agg == "count" and isinstance(result, pd.DataFrame):
                cols_lower = [c.lower() for c in result.columns]
                if not any(c in cols_lower for c in ["count", "counts", "n", "num"]):
                    return False
            if agg in ("sum", "mean", "min", "max") and isinstance(result, pd.DataFrame):
                numeric_cols = result.select_dtypes(include=["number"]).columns
                if len(numeric_cols) == 0:
                    return False
            return True
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


