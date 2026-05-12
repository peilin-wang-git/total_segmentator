# MRI Multi-Organ Segmentation Framework

一个可直接运行的 MRI 多器官自动分割工程化框架。该项目基于 **TotalSegmentator** 的 `total_mr` 任务，提供从目录扫描、预处理、推理、后处理、可视化到结果汇总的完整流程。

---

## 1. 项目简介

本项目面向“输入一个目录，自动处理目录下所有 MRI 图像”的实际场景，支持递归扫描并批量处理多种医学影像格式。核心目标是：在当前可落地方案中，尽可能分割 MRI 中可识别的器官，并输出可审阅结果与运行日志。

---

## 2. 功能列表

- 递归扫描输入目录下 MRI 文件
- 支持格式：`.nii`、`.nii.gz`、`.mha`、`.nrrd`
- 自动执行：读取 → 预处理 → 推理 → 后处理 → 保存
- 每个病例输出：
  - 分割标签图（`segmentation.nii.gz`）
  - 标签映射（`labels.json`）
  - 叠加预览图（`preview_overlay.png`，可关闭）
- 全局输出：
  - 运行日志（`run.log`）
  - 处理状态汇总（`summary.json`、`summary.csv`）
- 错误鲁棒性：单文件失败不影响批处理继续执行
- 支持 `--dry-run`（只扫描/校验，不推理）

---

## 3. 项目结构

```text
.
├── config/
│   └── example_config.yaml
├── mri_seg_framework/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── inference.py
│   ├── io_utils.py
│   ├── logging_utils.py
│   ├── pipeline.py
│   ├── postprocessing.py
│   ├── preprocessing.py
│   └── visualization.py
├── scripts/
│   └── download_models.py
├── README.md
└── requirements.txt
```

---

## 4. 安装方法

### 4.1 创建环境（推荐）

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4.2 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：`TotalSegmentator` 依赖深度学习运行时，首次安装/运行可能较慢。

---

## 5. 依赖说明

核心依赖见 `requirements.txt`：

- `TotalSegmentator`：多器官分割核心
- `SimpleITK`：医学图像 I/O
- `nibabel`：NIfTI 生态支持
- `numpy` / `pandas` / `matplotlib`：数据与可视化
- `PyYAML`：配置文件解析

---

## 6. 模型 / 权重准备

### 方案 A（推荐）
直接运行主程序，TotalSegmentator 通常会在首次推理时自动下载权重。

### 方案 B（提前下载）

```bash
python scripts/download_models.py --task total_mr
```

该脚本会触发一次 warm-up 推理，通常会将权重缓存到用户目录（例如 `~/.totalsegmentator`）。

---

## 7. 命令行使用示例

### 7.1 最小运行

```bash
python -m mri_seg_framework.cli \
  --input-dir ./data/input_mri \
  --output-dir ./data/output_seg
```

### 7.2 使用配置文件

```bash
python -m mri_seg_framework.cli --config config/example_config.yaml
```

### 7.3 仅做流程检查（不推理）

```bash
python -m mri_seg_framework.cli \
  --input-dir ./data/input_mri \
  --output-dir ./data/output_seg \
  --dry-run
```

### 7.4 关闭预览图

```bash
python -m mri_seg_framework.cli \
  --input-dir ./data/input_mri \
  --output-dir ./data/output_seg \
  --no-preview
```

### 7.5 使用 CSV（绝对路径）加载数据

当数据分散在不同目录时，可通过 CSV 第一列提供图像**绝对路径**：

```csv
image_path
/data/site_a/case001.nii.gz
/data/site_b/patient_12/scan.mha
```

运行示例：

```bash
python -m mri_seg_framework.cli \
  --input-csv /abs/path/to/images.csv \
  --device gpu \
  --gpu-id 1 \
  --intensity-norm zscore \
  --output-suffix _totalseg
```

说明：

- CSV 默认读取第一列作为输入路径；
- 路径必须是绝对路径；
- 仅处理支持的医学影像后缀（`.nii`、`.nii.gz`、`.mha`、`.nrrd`）。
- 可通过 `--output-suffix` 指定输出分割文件后缀（默认 `_seg`，例如 `_totalseg`）。
- 使用 `--input-csv` 时，`--output-dir` 可省略；省略后默认使用 `CSV所在目录/seg_run_outputs` 保存 `run.log`、`summary.json`、`summary.csv`、临时文件等运行产物。
- 可通过 `--device` 选择推理设备（`gpu` 或 `cpu`，默认 `gpu`）。
- 当 `--device gpu` 时，可通过 `--gpu-id` 指定使用第几块 GPU（默认 `0`）。
- 可通过 `--intensity-norm` 指定推理前强度标准化方式：
  - `none`（默认）：不做标准化；
  - `zscore`：均值方差标准化；
  - `percentile_minmax`：分位数裁剪后映射到 [0,1]；
  - `zscore_robust`：基于 median/MAD 的鲁棒标准化；
  - `itksnap_window`：按 case 自适应窗宽窗位（基于 Otsu 前景 + 对比度统计）后映射到 [0,1]。

---

## 8. 输入输出说明

### 输入

- 支持两种输入方式：
  1. `--input-dir`：目录递归扫描；
  2. `--input-csv`：CSV 第一列为绝对路径。
- 支持扩展名：`.nii`、`.nii.gz`、`.mha`、`.nrrd`。

### 输出

在 `output_dir` 下：

- `run.log`: 全局运行日志
- `summary.json` / `summary.csv`: 每病例状态汇总
- `<case_id>/segmentation.nii.gz`: 分割标签图
- `<case_id>/labels.json`: 标签值与器官名称映射
- `<case_id>/preview_overlay.png`: 叠加预览图（可选）
- `<case_id>/normalized_input.nii.gz`: 标准化后的推理输入图像（用于质检）
- `<case_id>/normalized_intensity_colorbar.png`: 标准化图像切片 + intensity colorbar（用于检查对比度）

- 使用 `--input-dir` 时：
  - `output_dir` 下生成 `run.log`、`summary.json/csv`、以及每个 `<case_id>/` 子目录输出。
- 使用 `--input-csv` 时：
  - 每个病例输出目录直接保存在该图像输入路径下，目录名为 `原文件名_totalseg`（例如 `/a/b/case1.nii.gz -> /a/b/case1_totalseg/`），其中包含 `segmentation.nii.gz`、`labels.json`、`preview_overlay.png`（可选）。
- 分割标签图会保存到原图同目录，文件名格式：`原文件名<output_suffix>.原后缀`。
  - 默认：`case1.nii.gz -> case1_seg.nii.gz`
  - 自定义：`--output-suffix _totalseg` 时为 `case1_totalseg.nii.gz`

- 使用 `--input-dir` 时：
  - `output_dir` 下生成 `run.log`、`summary.json/csv`、以及每个 `<case_id>/` 子目录输出。
- 使用 `--input-csv` 时：
  - 每个病例输出目录直接保存在该图像输入路径下，目录名为 `原文件名_totalseg`（例如 `/a/b/case1.nii.gz -> /a/b/case1_totalseg/`），其中包含 `segmentation.nii.gz`、`labels.json`、`preview_overlay.png`（可选）。
- 分割标签图会保存到原图同目录，文件名格式：`原文件名<output_suffix>.原后缀`。
  - 默认：`case1.nii.gz -> case1_seg.nii.gz`
  - 自定义：`--output-suffix _totalseg` 时为 `case1_totalseg.nii.gz`

---

## 9. 常见问题（FAQ）

### Q1: 为什么首次运行很慢？
A: 首次运行通常会下载模型权重并初始化深度学习环境。

### Q2: 某个文件失败会不会中断全任务？
A: 不会。框架会记录错误并继续处理其他文件。

### Q3: 能保证“全部器官”100%分割吗？
A: 不能。医学分割受模态、扫描范围、伪影、病灶、分辨率和模型泛化能力影响。

---

## 10. 已知限制

1. 本项目当前基于 TotalSegmentator `total_mr` 任务。
2. “可分割器官范围”受该任务定义与权重能力限制，非任意 MRI 都可分割全部解剖结构。
3. 非标准 MRI（极低信噪比、严重运动伪影、截断扫描范围）可能失败或结果较差。
4. 本项目提供的是**工程化批处理框架**，非医疗器械，不可直接用于临床诊断决策。

---

## 11. 最小可运行示例

假设目录结构：

```text
data/
└── input_mri/
    ├── case1.nii.gz
    └── patientA/
        └── scan.mha
```

运行：

```bash
python -m mri_seg_framework.cli --input-dir data/input_mri --output-dir data/output_seg
```

查看结果：

```bash
cat data/output_seg/summary.csv
```

---

## 支持范围与方案说明（重要）

- **支持 MRI 类型/模态**：以 TotalSegmentator `total_mr` 支持的 MRI 场景为主。
- **支持器官范围**：由 `total_mr` 任务对应标签定义决定，标签映射输出在每个病例 `labels.json` 中。
- **模型与权重来源**：`TotalSegmentator` 开源项目（运行时自动下载权重）。
- **已知失败场景**：输入损坏、维度异常、非 MRI/非目标解剖区域、极端伪影等。
