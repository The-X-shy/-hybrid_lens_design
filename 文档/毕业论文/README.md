# 毕业论文文件夹

- 原始 Word 模板：`北京理工大学本科生毕业设计（论文）模板（2023年12月）.docx`
- 自动转换备份：`latex_template_backup/bit_thesis_template_backup.tex`
- 严格 BIT 样式版（推荐起稿）：`latex_strict_bit/main.tex`

## 严格 BIT 样式版特性

- A4 版式，页边距按模板提取值设置：左 `3.00cm`、右 `2.60cm`、上 `3.50cm`、下 `2.60cm`
- 正文：小四、首行缩进 2 字符、基线按 22pt 设定
- 标题：一级/二级/三级标题按黑体与字号分级
- 图表题注：宋体五号、居中
- 封面、声明、中英文摘要、目录、正文、结论、附录、致谢结构齐全

## 编译

在 `latex_strict_bit` 目录执行：

```bash
xelatex main.tex
xelatex main.tex
```
