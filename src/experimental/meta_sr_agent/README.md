# 符号回归上下文搜索系统设计手册

## 1. 系统目标

给定目标公式 `target_formula` 和由其生成的数据集 `dataset`，搜索一组最佳 Claim。

这些 Claim 应满足：

```text
1. 对目标公式为真；
2. 能够直接根据 X-y 数据验证；
3. 能帮助公式生成器生成接近目标公式的候选公式。
```

系统最终输出：

```json
{
  "target_formula": "...",
  "dataset_id": "...",
  "best_record": {
    "claim_list_score": 0.0,
    "claim_list": [],
    "generated_equations": {
      "equation_list": [],
      "detail": "..."
    },
    "truth_evaluation": {
      "score": 0.0,
      "detail": "..."
    },
    "verifiability_evaluation": {
      "score": 0.0,
      "detail": "..."
    },
    "effectiveness_evaluation": {
      "score": 0.0,
      "detail": "..."
    }
  }
}
```

---

# 2. 核心对象

## 2.1 Claim

Claim 是一条关于目标函数输入输出行为的可验证事实。

字段：

```json
{
  "id": "str",
  "tool": "str",
  "claim": "str"
}
```

示例：

```json
{
  "id": "c1",
  "tool": "periodicity_probe",
  "claim": "当其他变量固定时，目标值关于 x1 呈周期性变化"
}
```

---

## 2.2 ClaimList

上下文就是一个 `List[Claim]`。

示例：

```json
[
  {
    "id": "c1",
    "tool": "additive_component_probe",
    "claim": "目标函数包含一个只依赖 x3 的加性分量，且该分量近似二次型"
  },
  {
    "id": "c2",
    "tool": "multiplicative_separability_probe",
    "claim": "移除 x3 相关分量后，剩余部分关于 x1 和 x2 近似乘性可分离"
  },
  {
    "id": "c3",
    "tool": "periodicity_probe",
    "claim": "剩余函数中关于 x1 的单变量模块具有周期性"
  },
  {
    "id": "c4",
    "tool": "log_linear_probe",
    "claim": "剩余函数中关于 x2 的单变量模块在 log 变换后近似线性"
  }
]
```

---

# 3. Agent 设计

## 3.1 ContextConstructor

### 宗旨

根据目标公式设计一组可直接从数据验证、且能帮助公式生成的 Claim。

### 任务

* 你知道目标公式，但不能直接泄露公式。
* 你需要生成一个 `List[Claim]`。
* 每条 Claim 必须描述目标函数的输入输出行为。
* 每条 Claim 必须能够由某种数据分析工具直接根据 X-y 数据验证。
* 每条 Claim 必须对公式生成有帮助，能够缩小公式搜索空间。
* 避免生成过于平凡的 Claim，例如“y 不是常数”“x1 与 y 有关系”。
* 避免直接复述目标公式中的子表达式。
* Claim 应尽量对应未来可以开发的工具。
* 可以生成关于变量相关性、周期性、单调性、对称性、多项式阶数、指数趋势、对数趋势、幂律趋势、加性分解、乘性可分离、低秩交互、残差结构、奇点、渐近线、有界性等事实。

### 输入示范

```json
{
  "target_formula": "sin(x1) * exp(x2) + x3^2",
  "previous_records": [
    {
      "claim_list_score": 0.42,
      "claim_list": [
        {
          "id": "c1",
          "tool": "periodicity_probe",
          "claim": "目标函数关于 x1 呈周期性"
        },
        {
          "id": "c2",
          "tool": "expression_probe",
          "claim": "目标函数包含 exp(x2) 结构"
        }
      ],
      "generated_equations": {
        "equation_list": [
          "sin(x1) + exp(x2) + x3^2"
        ],
        "detail": "公式生成器识别了主要模块，但没有恢复 x1 与 x2 的乘性关系。"
      },
      "truth_evaluation": { "score": 0.9, "detail": "两个 Claim 基本符合目标公式。"},
      "verifiability_evaluation": { "score": 0.45, "detail": "c2 过于接近直接说出 exp(x2)，不是一个容易直接从数据验证的行为事实。" },
      "effectiveness_evaluation": { "score": 0.55, "detail": "生成公式包含 sin、exp 和 x3^2，但错过乘性结构。" }
    }
  ]
}
```

### 输出示范

```json
{
  "claim_list": [
    {
      "id": "c1",
      "tool": "additive_component_probe",
      "claim": "目标函数包含一个只依赖 x3 的加性分量，且该分量近似二次型"
    },
    {
      "id": "c2",
      "tool": "multiplicative_separability_probe",
      "claim": "移除 x3 相关分量后，剩余函数关于 x1 和 x2 近似乘性可分离"
    },
    {
      "id": "c3",
      "tool": "periodicity_probe",
      "claim": "剩余函数中关于 x1 的单变量模块具有周期性"
    },
    {
      "id": "c4",
      "tool": "log_linear_probe",
      "claim": "剩余函数中关于 x2 的单变量模块在 log 变换后近似线性"
    }
  ]
}
```

---

## 3.2 FormulaGenerator

### 宗旨

根据 ClaimList 生成若干可能的公式。

### 任务

* 输入是变量名和 `List[Claim]`。
* 不读取目标公式。
* 将 Claim 视为之前工具探索得到的结果。
* 生成 K 个候选公式。
* 候选公式应尽量简单、可解释。
* 允许给出多个结构相近的变体。

### 输入示范

```json
{
  "variables": ["x1", "x2", "x3"],
  "claim_list": [
    {
      "id": "c1",
      "tool": "additive_component_probe",
      "claim": "目标函数包含一个只依赖 x3 的加性分量，且该分量近似二次型"
    },
    {
      "id": "c2",
      "tool": "multiplicative_separability_probe",
      "claim": "移除 x3 相关分量后，剩余函数关于 x1 和 x2 近似乘性可分离"
    }
  ],
  "num_candidates": 5
}
```

### 输出示范

```json
{
  "equation_list": [
    "sin(x1) * exp(x2) + x3^2",
    "cos(x1) * exp(x2) + x3^2",
    "(a * sin(x1) + b * cos(x1)) * exp(x2) + c * x3^2"
  ],
  "detail": "根据 Claim，公式应包含 x3 的二次加性项，以及 x1 周期模块和 x2 指数型模块的乘积。"
}
```

---

## 3.3 FormulaEvaluator

### 宗旨

评估候选公式与目标公式的接近程度，并保持多次评估偏好一致。

### 任务

* 输入是目标公式、候选公式、变量名和之前的公式评估记录。
* 比较每个候选公式与目标公式的结构接近程度。
* 为每个候选公式打分。
* 参考之前的 Formula Evaluation，保持评分尺度和偏好一致。
* 可以考虑变量依赖、运算符、组合结构、可分离结构、等价变换。
* 最终 `effectiveness.score` 使用所有候选公式分数中的最大值。

### 输入示范

```json
{
  "target_formula": "sin(x1) * exp(x2) + x3^2",
  "equation_list": [
    "sin(x1) * exp(x2) + x3^2",
    "cos(x1) * exp(x2) + x3^2",
    "sin(x1) + exp(x2) + x3^2"
  ],
  "variables": ["x1", "x2", "x3"],
  "previous_formula_evaluations": [
    {
      "target_formula": "sin(x1) * exp(x2) + x3^2",
      "candidate_equation": "sin(x1) + exp(x2) + x3^2",
      "score": 0.55,
      "detail": "识别了三个主要模块，但错过 x1 与 x2 的乘性耦合。"
    }
  ]
}
```

### 输出示范

```json
{
  "candidate_scores": [
    {
      "equation": "sin(x1) * exp(x2) + x3^2",
      "score": 0.97,
      "detail": "结构基本匹配目标公式。"
    },
    {
      "equation": "cos(x1) * exp(x2) + x3^2",
      "score": 0.75,
      "detail": "整体结构正确，但周期模块形式不同。"
    },
    {
      "equation": "sin(x1) + exp(x2) + x3^2",
      "score": 0.55,
      "detail": "识别了主要模块，但错过 x1 与 x2 的乘性关系。"
    }
  ]
}
```

---

## 3.4 TruthEvaluator

### 宗旨

判断 ClaimList 中最不符合目标公式的 Claim，并据此评价整组 Claim 的真实性。

### 任务

* 输入是目标公式和 `List[Claim]`。
* 逐条判断 Claim 是否为目标公式的真实性质。
* 评估每条 Claim 的 truth 分数。
* 指出错误、部分正确、表述过强的 Claim。
* 可以进行多轮对话。
* 可以调用 `code_executor`，通过符号化或数值采样辅助判断。
* 最终 `truth.score` 使用所有 Claim 分数中的最小值。

### 输入示范

```json
{
  "target_formula": "sin(x1) * exp(x2) + x3^2",
  "claim_list": [
    {
      "id": "c1",
      "tool": "additive_component_probe",
      "claim": "目标函数包含一个只依赖 x3 的加性二次分量"
    },
    {
      "id": "c2",
      "tool": "additive_separability_probe",
      "claim": "目标函数关于 x1 和 x2 是加性可分离的"
    }
  ]
}
```

### 输出示范

```json
{
  "score": 0.2,
  "detail": "c1 的真实性较高，因为 x3^2 是只依赖 x3 的加性二次分量。c2 的真实性较低，因为目标公式中 x1 和 x2 通过 sin(x1) * exp(x2) 发生乘性耦合，不是加性可分离。逐条分数：c1=0.95，c2=0.2。因此整体 truth score 取最小值 0.2。"
}
```

---

## 3.5 VerifiabilityEvaluator

### 宗旨

判断 ClaimList 中最难仅基于 X-y 数据验证的 Claim，并据此评价整组 Claim 的可验证性。

### 任务

* 输入是数据引用、变量名和 `List[Claim]`。
* 不依赖目标公式本身。
* 逐条判断 Claim 是否有希望被工具从数据中发现。
* 评估每条 Claim 的 verifiability 分数。
* 指出无法验证、难以验证、需要强假设的 Claim。
* 可以进行多轮对话。
* 可以调用 `code_executor`，写临时代码对数据做简单探索。
* 不要求真正实现完整工具，只需要判断验证前景。
* 最终 `verifiability.score` 使用所有 Claim 分数中的最小值。

### 输入示范

```json
{
  "data_reference": "dataset_001",
  "variables": ["x1", "x2", "x3"],
  "claim_list": [
    {
      "id": "c1",
      "tool": "periodicity_probe",
      "claim": "目标函数关于 x1 具有周期性"
    },
    {
      "id": "c2",
      "tool": "unknown",
      "claim": "该公式具有相对论含义"
    }
  ],
  "available_tools": [
    "code_executor",
    "slice_sampler",
    "curve_fit_probe",
    "rank1_probe"
  ]
}
```

### 输出示范

```json
{
  "score": 0.0,
  "detail": "c1 可以通过固定其他变量后对 x1 做切片采样，并检测自相关或频谱峰来验证，估计分数为 0.75。c2 是语义或物理解释，无法仅从 X-y 数据稳定推出，估计分数为 0.0。因此整体 verifiability score 取最小值 0.0。"
}
```

---

# 4. 指标定义

## 4.1 Truth Score

衡量整组 Claim 的真实性。

每条 Claim 的分数范围：

```text
0.0 = 明显错误
0.5 = 部分正确或表述过强
1.0 = 明确正确
```

整体分数：

```text
truth.score = min(per_claim_truth_scores)
```

---

## 4.2 Verifiability Score

衡量整组 Claim 是否能直接从 X-y 数据中被验证。

每条 Claim 的分数范围：

```text
0.0 = 无法从数据验证
0.5 = 需要较强假设或复杂工具
1.0 = 可以由明确工具稳定验证
```

整体分数：

```text
verifiability.score = min(per_claim_verifiability_scores)
```

---

## 4.3 Effectiveness Score

衡量公式生成器产生的候选公式与目标公式的接近程度。

每个候选公式的分数范围：

```text
0.0 = 完全不相关
0.5 = 捕捉部分变量或部分结构
1.0 = 与目标公式等价或近似等价
```

整体分数：

```text
effectiveness.score = max(candidate_equation_scores)
```

---

## 4.4 Claim List Score

整组 Claim 的最终分数。

默认公式：

```text
claim_list_score =
    truth.score
    * verifiability.score
    * effectiveness.score
    - length_penalty
```

其中：

```text
length_penalty = 0.02 * len(claim_list)
```

---

# 5. 主循环伪代码（仅供参考，如有不同，以实际代码为准）

```python
def search_claim_lists(
    target_formula,
    dataset,
    dataset_id,
    variables,
    max_iterations,
    num_claim_lists_per_iter,
    num_equations_per_claim_list,
):
    archive = []
    previous_formula_evaluations = []

    for iteration in range(max_iterations):

        previous_records = sample_archive_records(archive)

        proposals = ContextConstructor.generate(
            target_formula=target_formula,
            previous_records=previous_records,
            n=num_claim_lists_per_iter,
        )

        for proposal in proposals:
            claim_list = proposal["claim_list"]

            generated_equations = FormulaGenerator.generate(
                variables=variables,
                claim_list=claim_list,
                k=num_equations_per_claim_list,
            )

            formula_eval = FormulaEvaluator.evaluate(
                target_formula=target_formula,
                equation_list=generated_equations["equation_list"],
                variables=variables,
                previous_formula_evaluations=previous_formula_evaluations,
            )

            previous_formula_evaluations.extend(
                [
                    {
                        "target_formula": target_formula,
                        "candidate_equation": item["equation"],
                        "score": item["score"],
                        "detail": item["detail"],
                    }
                    for item in formula_eval["candidate_scores"]
                ]
            )

            effectiveness_score = max(
                item["score"] for item in formula_eval["candidate_scores"]
            )

            truth_eval = TruthEvaluator.evaluate(
                target_formula=target_formula,
                claim_list=claim_list,
                tools=["code_executor"],
            )

            verifiability_eval = VerifiabilityEvaluator.evaluate(
                dataset=dataset,
                variables=variables,
                claim_list=claim_list,
                tools=["code_executor"],
            )

            truth_score = truth_eval["score"]
            verifiability_score = verifiability_eval["score"]

            claim_list_score = compute_claim_list_score(
                truth_score=truth_score,
                verifiability_score=verifiability_score,
                effectiveness_score=effectiveness_score,
                num_claims=len(claim_list),
            )

            record = {
                "claim_list_score": claim_list_score,
                "claim_list": claim_list,
                "generated_equations": generated_equations,
                "truth_evaluation": { "score": truth_score, "detail": truth_eval["detail"], },
                "verifiability_evaluation": { "score": verifiability_score, "detail": verifiability_eval["detail"], },
                "effectiveness_evaluation": { "score": effectiveness_score, "detail": formula_eval["candidate_scores"], },
            }

            archive.append(record)

    best_record = max(
        archive,
        key=lambda r: r["claim_list_score"],
    )

    return {
        "target_formula": target_formula,
        "dataset_id": dataset_id,
        "best_record": best_record,
    }
```

---

# 6. Archive Record 格式

```json
{
  "claim_list_score": 0.69,
  "claim_list": [
    {
      "id": "c1",
      "tool": "additive_component_probe",
      "claim": "目标函数包含一个只依赖 x3 的加性分量，且该分量近似二次型"
    },
    {
      "id": "c2",
      "tool": "multiplicative_separability_probe",
      "claim": "移除 x3 相关分量后，剩余函数关于 x1 和 x2 近似乘性可分离"
    }
  ],
  "generated_equations": {
    "equation_list": [
      "sin(x1) * exp(x2) + x3^2",
      "cos(x1) * exp(x2) + x3^2"
    ],
    "detail": "公式生成器根据 Claim 生成了周期模块、指数模块、二次模块的组合。"
  },
  "truth_evaluation": { 
    "score": 0.95, 
    "detail": "所有 Claim 基本符合目标公式。" 
  },
  "verifiability_evaluation": { 
    "score": 0.75, 
    "detail": "所有 Claim 都有可行的数据验证路径，其中乘性可分离检测需要较好的采样覆盖。" 
  },
  "effectiveness_evaluation": {
    "score": 0.97,
    "detail": [
      {
        "equation": "sin(x1) * exp(x2) + x3^2",
        "score": 0.97,
        "detail": "结构基本匹配目标公式。"
      },
      {
        "equation": "cos(x1) * exp(x2) + x3^2",
        "score": 0.75,
        "detail": "整体结构正确，但周期模块形式不同。"
      }
    ]
  }
  }
```

---

# 7. 最终输出格式

```json
{
  "target_formula": "sin(x1) * exp(x2) + x3^2",
  "dataset_id": "dataset_001",
  "best_record": {
    "claim_list_score": 0.69,
    "claim_list": [
      {
        "id": "c1",
        "tool": "additive_component_probe",
        "claim": "目标函数包含一个只依赖 x3 的加性分量，且该分量近似二次型"
      },
      {
        "id": "c2",
        "tool": "multiplicative_separability_probe",
        "claim": "移除 x3 相关分量后，剩余函数关于 x1 和 x2 近似乘性可分离"
      }
    ],
    "generated_equations": {
      "equation_list": [
        "sin(x1) * exp(x2) + x3^2",
        "cos(x1) * exp(x2) + x3^2"
      ],
      "detail": "公式生成器根据 Claim 生成了周期模块、指数模块、二次模块的组合。"
    },
    "truth_evaluation": {
      "score": 0.95,
      "detail": "所有 Claim 基本符合目标公式。"
    },
    "verifiability_evaluation": {
      "score": 0.75,
      "detail": "所有 Claim 都有可行的数据验证路径，其中乘性可分离检测需要较好的采样覆盖。"
    },
    "effectiveness_evaluation": {
      "score": 0.97,
      "detail": [
        {
          "equation": "sin(x1) * exp(x2) + x3^2",
          "score": 0.97,
          "detail": "结构基本匹配目标公式。"
        },
        {
          "equation": "cos(x1) * exp(x2) + x3^2",
          "score": 0.75,
          "detail": "整体结构正确，但周期模块形式不同。"
        }
      ]
    }
  }
}
```
