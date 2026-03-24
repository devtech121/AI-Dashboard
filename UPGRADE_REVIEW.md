# Upgrade Review and Code Audit

Date: 2026-03-24
Project: AI Data Analyst Dashboard

## Summary
This document captures:
- Bugs and errors found in the current codebase
- Comment cleanup candidates (non-explanatory or issue-tracking comments)
- Upgrade proposals (requires approval before implementation)
- Pipeline assessment (data flow and processing steps)

## Bugs and Errors Found
1. Correlation heatmap is never built due to mis-indented and unreachable code.
   - File: modules/chart_generator.py
   - Details: _build_correlation_heatmap() returns early without building a figure, and the heatmap logic appears after a return statement inside _suggest_semantic_charts(), making it unreachable.

2. README documentation is outdated and references HuggingFace usage and model fallback that no longer match the Groq/Anthropic implementation.
   - File: README.md

## Comment Cleanup Candidates
The following comments are tracking issues or historical notes rather than describing behavior. They should be removed or replaced with concise explanatory comments where needed.
- app.py: multiple "Issue #" comments in initialize_session_state(), process_uploaded_file(), render_filters(), render_chat_section().
- modules/ai_engine.py: "Issue #" comments in class docstring, cache comment, and _build_analysis_prompt section headers.
- modules/anomaly_detector.py: class docstring issue note.
- modules/chart_generator.py: issue note in module docstring.
- modules/profiler.py: class docstring issue note.
- modules/chat_engine.py: issue note block comment at top (some of this is still relevant, but should be converted into concise security rationale rather than issue history).

## Upgrade Proposals (Require Approval)
1. Fix and enhance correlation heatmap generation.
   - Restore _build_correlation_heatmap implementation and ensure it is called correctly.
   - Add safe fallback if numeric columns are insufficient.

2. Improve chat execution sandbox termination.
   - Replace threading timeout with a subprocess or multiprocessing-based hard kill to prevent runaway CPU usage.

3. Add structured pipeline timing and telemetry (local-only).
   - Capture step timings (load, profile, anomaly, AI, KPIs, charts, insights) for diagnostics in the UI.

4. Add AI response schema validation with JSON Schema.
   - Validate AI output (kpis/charts/insights) before passing to generators.

5. Update README and configuration docs to match Groq/Anthropic models and API key setup.

6. Optional: Add cached profile/anomaly results scoped by session and dataset hash to avoid recomputation within a session.

## Pipeline Assessment
Current pipeline in app.py (process_uploaded_file):
1. Load and preprocess CSV (DataLoader).
2. Profile dataset (DataProfiler).
3. Detect anomalies (AnomalyDetector).
4. AI analysis with fallback (AIEngine).
5. Verify/repair AI output (AnalysisVerifier).
6. Generate KPIs (KPIGenerator).
7. Generate charts (ChartGenerator).
8. Generate insights (InsightsGenerator).

Status:
- Overall flow is coherent and ordered correctly.
- Verified analysis step properly sits between AI analysis and KPI/Chart generation.
- Primary pipeline issue: correlation heatmap step is broken due to the bug noted above.

## Notes
- No unit tests were found in the repository.
- Some emojis appear as mojibake in terminal output; verify file encoding is UTF-8 if this renders incorrectly in the UI.
