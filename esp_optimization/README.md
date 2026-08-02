# 水泥电除尘器协同优化：复现工程与正式论文

本目录保存 A 题四问的数值解答、模型代码、结果表、图形与正式论文。仓库根目录同时提供原始题面和比赛数据，程序以只读方式加载 `Cement_ESP_Data.csv`。

## 主要结论

- `C_out=50` 呈现显著边界堆积，本文将其作为疑似仪表上限删失进行建模，不做比例换算，并单独报告该假设的适用边界。
- 原始排放标签无法识别控制效应；排放外推使用删失运行点锚定的物理信息模型，并单独报告先验敏感性。
- 能耗模型采用 `U_i²` 与 `1/T_i`，全样本解释度约 99.8%，逐日留一 RMSE 约 6 kW。
- 10 mg/Nm³在历史操作边界内可行；5 mg/Nm³有部分工况在原边界内不可行。按1%步长筛查后，电压上界放宽2%是首个全工况可行情景，3%仅作为容量裕量敏感性。

详细结果见 [answers/逐问解答.md](answers/逐问解答.md)，正式论文源码见 [paper/main.tex](paper/main.tex)。

## 数据说明

原始 `A题.docx` 与 `Cement_ESP_Data.csv` 位于仓库上一级目录：

```text
workspace/
|-- Cement_ESP_Data.csv
`-- esp_optimization/
```

也可以修改 `config/model.yaml` 中的 `project.input_csv` 指向其他本地数据文件。

## 环境与运行

建议使用 Python 3.11：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_all.py
.\.venv\Scripts\python.exe scripts\audit_regime_settings.py
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

编译正式论文（需安装 Tectonic）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_paper.ps1
```

正式论文输出到 `output/pdf/Cement_ESP_Optimization_Paper_Final.pdf`，编译辅助文件保存在 `build/latex/`，不污染论文源码目录。

所有随机过程固定基准种子 2026。模型口径、机会约束、先验和优化参数均在 `config/model.yaml` 中，不需要修改源码。

## 目录

- `src/`：数据审计、删失排放数字孪生、能耗模型、工况识别、优化和制图。
- `config/model.yaml`：唯一配置入口。
- `scripts/run_all.py`：端到端运行入口。
- `answers/逐问解答.md`：四问推导与数值结论。
- `outputs/tables/`：可直接引用到论文的 UTF-8 CSV 表。
- `outputs/figures/`：11 幅论文级 PNG 图。
- `outputs/models/`、`outputs/logs/`：模型摘要和数据审计日志。
- `tests/`：关键不变量、模型方向和结果验收测试。
- `paper/`：A4 `ctexart` 正式论文、分章节源码与参考文献；`paper/reference_materials/` 单独归档公开原文与 DOI 元数据。
- `build/latex/`：LaTeX 编译辅助文件。
- `output/pdf/`：最终正式竞赛论文 PDF。

## 结果解释边界

附件未提供二次电流、粉尘比电阻、粒径分布和实际振打触发时刻，且出口数据严重右删失。因此，10/5 mg/Nm³的排放结果不能被表述为附件数据的直接实证预测；它们是有来源、可审计并带敏感性区间的工程外推。现场应用前必须重新标定。
