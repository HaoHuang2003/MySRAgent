# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Deterministic mock backend for the experimental meta SR agent."""

from __future__ import annotations

import re

from experimental.meta_sr_agent.schema import Claim


def mock_claim_lists(
    target_formula: str,
    variables: list[str],
    max_claim_list_length: int,
    local_sample_size: int,
) -> list[list[Claim]]:
    rhs = rhs_of(target_formula)
    claims: list[Claim] = []
    next_id = 1

    for variable in variables:
        if re.search(rf"\b(sin|cos|tan)\s*\([^)]*\b{re.escape(variable)}\b", rhs):
            claims.append(
                Claim(
                    id=f"c{next_id}",
                    tool_name="periodicity_probe",
                    claim=f"目标函数在包含 {variable} 的方向上呈周期性变化",
                )
            )
            next_id += 1
        if re.search(rf"\bexp\s*\([^)]*\b{re.escape(variable)}\b", rhs):
            claims.append(
                Claim(
                    id=f"c{next_id}",
                    tool_name="log_linear_probe",
                    claim=f"目标函数中与 {variable} 相关的正值变化在对数变换后近似线性",
                )
            )
            next_id += 1
        compact_rhs = rhs.replace(" ", "")
        if re.search(rf"\b{re.escape(variable)}\s*(\*\*|\^)\s*2\b", rhs) or f"{variable}*{variable}" in compact_rhs:
            claims.append(
                Claim(
                    id=f"c{next_id}",
                    tool_name="polynomial_degree_probe",
                    claim=f"目标函数包含关于 {variable} 的近似二次型变化",
                )
            )
            next_id += 1
        if variable in rhs and not any(variable in claim.claim for claim in claims):
            claims.append(
                Claim(
                    id=f"c{next_id}",
                    tool_name="sensitivity_probe",
                    claim=f"目标函数对 {variable} 的局部变化有稳定响应",
                )
            )
            next_id += 1

    if "*" in rhs and len(variables) >= 2:
        claims.append(
            Claim(
                id=f"c{next_id}",
                tool_name="multiplicative_separability_probe",
                claim=f"目标函数在 {variables[:2]} 上存在可检测的乘性耦合或低秩交互",
            )
        )
        next_id += 1

    if "+" in rhs or "-" in rhs:
        claims.append(
            Claim(
                id=f"c{next_id}",
                tool_name="additive_component_probe",
                claim="目标函数可以分解为少量可检测的加性行为模块",
            )
        )

    if not claims:
        claims.append(
            Claim(
                id="c1",
                tool_name="smoothness_probe",
                claim="目标函数在采样区域内表现为平滑的低复杂度函数",
            )
        )

    base = claims[:max_claim_list_length]
    compact = claims[: max(1, min(3, max_claim_list_length))]
    rotated = claims[1 : max_claim_list_length + 1] or compact
    return [base, compact, rotated][:local_sample_size]


def mock_equations(
    variables: list[str],
    claim_list: list[Claim],
    target_formula: str,
) -> list[str]:
    rhs = rhs_of(target_formula)
    equations = [rhs]
    first = variables[0] if variables else "x1"
    second = variables[1] if len(variables) > 1 else first
    claim_text = " ".join(claim.claim for claim in claim_list)

    if "周期" in claim_text and "对数" in claim_text:
        equations.append(f"sin({first}) * exp({second})")
    if "周期" in claim_text:
        equations.extend([f"sin({first})", f"cos({first})"])
    if "二次" in claim_text:
        equations.append(" + ".join(f"{v}**2" for v in variables))
    if "加性" in claim_text and variables:
        equations.append(" + ".join(variables))
    if len(variables) >= 2:
        equations.append(f"sin({variables[0]} - {variables[1]})")
        equations.append(f"{variables[0]} * {variables[1]}")

    deduped = []
    for value in equations:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def mock_truth_score(target_formula: str, claim: Claim) -> float:
    rhs = rhs_of(target_formula)
    text = claim.claim
    if "周期" in text:
        return 0.95 if re.search(r"\b(sin|cos|tan)\s*\(", rhs) else 0.25
    if "对数" in text or "正值" in text:
        return 0.9 if "exp(" in rhs or "log(" in rhs else 0.45
    if "二次" in text:
        return 0.9 if re.search(r"(\*\*|\^)\s*2\b", rhs) else 0.45
    if "乘性" in text or "耦合" in text:
        return 0.85 if "*" in rhs else 0.4
    if "加性" in text:
        return 0.8 if "+" in rhs or "-" in rhs else 0.55
    return 0.7


def mock_truth_detail(target_formula: str, claim: Claim) -> str:
    rhs = rhs_of(target_formula)
    text = claim.claim
    if "周期" in text:
        if re.search(r"\b(sin|cos|tan)\s*\(", rhs):
            return "目标公式包含三角函数结构，因此该周期性行为判断基本成立。"
        return "目标公式未显示明显三角周期结构，该 claim 真实性较弱。"
    if "对数" in text or "正值" in text:
        if "exp(" in rhs or "log(" in rhs:
            return "目标公式含指数或对数相关结构，该趋势描述有公式依据。"
        return "目标公式未显示指数或对数结构，该趋势描述可能过强。"
    if "二次" in text:
        if re.search(r"(\*\*|\^)\s*2\b", rhs):
            return "目标公式含平方项，因此二次型行为判断基本成立。"
        return "目标公式未显示平方项，该二次型描述只可能部分成立。"
    if "乘性" in text or "耦合" in text:
        if "*" in rhs:
            return "目标公式含乘法结构，因此存在乘性耦合的判断有依据。"
        return "目标公式没有明显乘法耦合，该 claim 真实性较弱。"
    if "加性" in text:
        if "+" in rhs or "-" in rhs:
            return "目标公式包含加减组合，可视作少量加性行为模块的组合。"
        return "目标公式没有明显加性分解，该 claim 只是宽泛近似。"
    return "该 claim 是宽泛行为描述，当前启发式只能给出中等真实性估计。"


def mock_discoverability_score(claim: Claim) -> float:
    tool_scores = {
        "periodicity_probe": 0.85,
        "log_linear_probe": 0.75,
        "polynomial_degree_probe": 0.85,
        "multiplicative_separability_probe": 0.7,
        "additive_component_probe": 0.7,
        "sensitivity_probe": 0.75,
        "smoothness_probe": 0.65,
    }
    return tool_scores.get(claim.tool_name, 0.55)


def mock_discoverability_detail(claim: Claim) -> str:
    tool_details = {
        "periodicity_probe": "可通过固定其他变量做单变量切片，并用自相关或频谱峰检测周期性。",
        "log_linear_probe": "可通过切片采样后做正值筛选和 log 变换，再检查线性拟合质量。",
        "polynomial_degree_probe": "可通过单变量或残差切片拟合低阶多项式并比较二次项贡献。",
        "multiplicative_separability_probe": "可在去除主效应后用秩一近似或交互矩阵检测乘性可分离性。",
        "additive_component_probe": "可用变量分组、残差拟合和加性模型比较验证，但需要较好的采样覆盖。",
        "sensitivity_probe": "可用局部扰动或梯度/差分估计检测变量响应。",
        "smoothness_probe": "可用局部邻域差分或简单回归残差检测平滑低复杂度行为。",
    }
    return tool_details.get(claim.tool_name, "该 claim 需要额外设计专门工具，可发现性暂按中等偏低估计。")


def mock_formula_score(target_formula: str, equation: str) -> float:
    target = normalize_formula(rhs_of(target_formula))
    candidate = normalize_formula(rhs_of(equation))
    if target == candidate:
        return 0.98
    target_tokens = set(re.findall(r"[A-Za-z_]\w*|\d+|\*\*|\^|[+\-*/()]", target))
    candidate_tokens = set(re.findall(r"[A-Za-z_]\w*|\d+|\*\*|\^|[+\-*/()]", candidate))
    if not target_tokens or not candidate_tokens:
        return 0.0
    overlap = len(target_tokens & candidate_tokens) / len(target_tokens | candidate_tokens)
    structure_bonus = 0.1 if any(op in target and op in candidate for op in ("sin", "cos", "exp", "log")) else 0.0
    return min(0.95, 0.15 + 0.75 * overlap + structure_bonus)


def rhs_of(formula: str) -> str:
    return formula.split("=", 1)[1].strip() if "=" in formula else formula.strip()


def normalize_formula(formula: str) -> str:
    return rhs_of(formula).replace(" ", "").replace("^", "**")
