# 水泥烧成系统电除尘器协同优化

本仓库包含数学建模 A 题原始题面、分钟级运行数据、完整建模代码、逐问解答、生成图表以及正式 LaTeX 论文。

## 仓库内容

- `A题.docx`：原始题面。
- `Cement_ESP_Data.csv`：连续 7 天分钟级运行数据。
- `esp_optimization/`：可复现代码、配置、测试、结果表、图形和论文源码。
- `esp_optimization/output/pdf/Cement_ESP_Optimization_Paper.pdf`：正式论文 PDF。

## 快速复现

```powershell
cd esp_optimization
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_all.py
.\.venv\Scripts\python.exe -m pytest -q
```

更完整的模型说明和编译步骤见 [`esp_optimization/README.md`](esp_optimization/README.md)。

