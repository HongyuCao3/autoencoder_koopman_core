# 任务：proposal 四张图按「单栏宽、字号 ≥ 8pt」重出（2026-11-23）

> 来源：`E:\Kunpeng Liu\ICLR2027\MFAI_materials_request_Hongyu.md` 第 1 项的格式要求（PDF 矢量、单栏宽度、字号不小于 8pt）。
> 现状：`proposal/figs/fig1–4.pdf`（11-05 出）量得正文字 6.2–7.0pt、刻度 4–7pt；Fig 2 / Fig 4 宽 6.5 in（双栏）。四张都不达标。
> 执行位置：**服务器**（Fig 4 读 `results/core_excite_reachability/`、`results/core_excite_b_identifiability/`、`results/b_identifiability/`，本地没有）。
> 零 GPU、只读 `paper/evidence/numbers.yaml` 与 `results/`，只写 `proposal/figs/`。不重算任何数字；`paper/` 不碰。

## 1. 改什么（`docs/article/proposal_figs/make_proposal_figs.py`）

### 1.1 加一个 `--layout {wide,single}` 开关，默认 `wide`（保持 11-05 产物不变）

`single` 时：
- 输出到 `proposal/figs/single/`（新目录），文件名不变；`_proposal_fig_report.json` 同样写到该目录。
- 全局 `rcParams`：`font.size 9`、`axes.labelsize 9`、`axes.titlesize 9`、`xtick/ytick.labelsize 8`、`legend.fontsize 8`。
- 脚本里所有硬编码的 `fontsize=5.8 / 6.0 / 6.2 / 6.3 / 6.6 / 7 / 7.2` 一律改为经 `FS(x)` 取值：`FS = lambda x: max(x, 8.0) if LAYOUT == "single" else x`。注释文字（"hollow = interval contains 0" 等）允许 8pt，不允许更小。
- 画布宽 **3.3 in**（单栏；Fig 1 例外见下）。

### 1.2 逐图

| 图 | wide（现状） | single 改法 |
|---|---|---|
| Fig 1 框架图 | 6.5 × 2.0 | **保持全宽**（它替换的是 p.2 的整行箭头式，F1 方案 B 就是全宽），只把字号抬到 ≥8：标题 9、框内正文 8、输入口标签 8。高度允许涨到 2.4 in。若框内文字放不下，缩短文字不缩字号（例如 "realized in preliminary work (Sec. 3.2)" → "preliminary work, Sec. 3.2"）。 |
| Fig 2 memory | 1×2，6.5 × 2.05 | 改 **2×1 上下叠放**，3.3 × 4.2：上 (a) 7 任务分组柱，x 轴任务名改两行或 30° 斜排、8pt；下 (b) 4 任务深度折线，任务名标签挪到图例（8pt）而不是线尾贴字，避免重叠。 |
| Fig 3 lifting | 3.2 × 2.0 | 宽改 3.3，高 2.3；y 轴任务名 8pt；顶部注释 8pt，放不下就拆成两行。 |
| Fig 4 prompt authority | 1×2，6.5 × 2.2 | 改 **2×1 上下叠放**，3.3 × 4.4：上 (a) 散点 + 拟合线 + 可达上界虚线，图例 8pt 两列；下 (b) 三任务 CB 点 + 区间，三个子轴的 x 刻度 8pt（现在 5.8），单位标签 8pt。 |

### 1.3 验收（写进脚本末尾，`single` 下不过则 `SystemExit(1)`）

```python
import pdfplumber
for pdf in sorted((OUT).glob("fig*.pdf")):
    p = pdfplumber.open(pdf).pages[0]
    w_in = p.width / 72
    sizes = [round(c["size"], 1) for c in p.chars if c["text"].strip()]
    small = [s for s in sizes if s < 8.0]
    print(pdf.name, f"{w_in:.2f} in", "min font", min(sizes), "n<8:", len(small))
    assert (w_in <= 3.4) or pdf.name.startswith("fig1"), pdf.name
    assert not small, f"{pdf.name}: {len(small)} glyphs below 8pt"
```

注：mathtext 的上下标会落到 6pt 左右；若只剩这类字符不过，把对应标签改成纯文本（`c_k^p` → `c (prompt)`），不放宽阈值。

## 2. 顺带：把 Umesh 要的原值拷回本地

```
scp <server>:<repo>/results/core_excite_b_identifiability/summary.json \
    <server>:<repo>/results/b_identifiability/{tsar_cefr,constraint}.json \
    <server>:<repo>/results/spectral_saturation/summary.json \
    <server>:<repo>/results/core_excite_reachability/summary.json \
    "E:\Kunpeng Liu\ICLR2027\MFAI_materials_2026-11-23\03_B_matrix\server_results\"
```

## 3. 完成 = 什么

- `proposal/figs/single/fig1–4.pdf` + `.png` + `_proposal_fig_report.json`；验收脚本通过。
- `proposal/figs/`（wide）逐字节不变。
- 四张 single 版复制到 `E:\Kunpeng Liu\ICLR2027\MFAI_materials_2026-11-23\01_prelim\single\`。
- LEDGER §十「零 GPU 记录」补一行：件 = 图重排（排版，不改数字）；判 = 验收过/不过；产物 = `proposal/figs/single/`。
- commit message 建议：`proposal figs: single-column layout, fonts >= 8pt (materials request 2026-11-23)`。

## 4. 不做

- 不加新数据、不改任何 id；不画 two-futures（清单里的 "Fig 3"）——描述性图、不配数字，且只对 9 月版段落有用。
- 不画 Table 2 柱状图（11-05 决定：无 seed、无日志）。
