from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "analysis_results"
DOCX_PATH = OUTPUT_DIR / "TwoStage_Stage1_Stage2_Algorithms.docx"
TEXT_PATH = OUTPUT_DIR / "TwoStage_Stage1_Stage2_Algorithms.txt"
STAGE1_FLOWCHART = OUTPUT_DIR / "TwoStage_Stage1_Flowchart.png"
STAGE2_FLOWCHART = OUTPUT_DIR / "TwoStage_Stage2_Flowchart.png"


STAGE1 = {
    "title": "算法 4.3 双阶段实验的一阶段（sim_pretrain）运行流程",
    "known": (
        "已知：系统配置文件：C_sys，训练配置文件：C_train，奖励配置文件：C_reward，"
        "阶段一训练轮数：E_1，单轮最大步数：T_1，批大小：N，折扣因子：γ，软更新系数：τ。"
    ),
    "goal": "求：仿真预训练模型 M_sim，阶段一训练日志 L_sim。",
    "lines": [
        {"num": 1, "text": "初始化演员网络参数 θ_μ 和评论家网络参数 θ_Q，初始化目标网络 θ_μ' ← θ_μ，θ_Q' ← θ_Q。", "indent": 0, "bold": False},
        {"num": 2, "text": "初始化经验重放缓冲区 D ← ∅，初始化探索噪声过程 ψ。", "indent": 0, "bold": False},
        {"num": 3, "text": "启动 Unity / AirSim，初始化多无人机算法服务器 Server 和训练环境 Env。", "indent": 0, "bold": False},
        {"num": 4, "text": "for episode = 1 to E_1 do", "indent": 0, "bold": True},
        {"num": 5, "text": "重置环境，获取初始状态 s_t，并初始化当前回合的电量、碰撞计数和扫描状态。", "indent": 1, "bold": False},
        {"num": 6, "text": "for t = 1 to T_1 do", "indent": 1, "bold": True},
        {"num": 7, "text": "将状态 s_t 输入演员网络，得到连续动作 a_t ← μ(s_t|θ_μ)。", "indent": 2, "bold": False},
        {"num": 8, "text": "给动作加入探索噪声，得到执行动作 ã_t ← a_t + ψ_t。", "indent": 2, "bold": False},
        {"num": 9, "text": "对 ã_t 进行裁剪、平滑和安全约束，并映射为 APF 参数 W_t ← {w_r, w_e, w_d, w_l, w_h, d_o, c_o}。", "indent": 2, "bold": False},
        {"num": 10, "text": "计算 F_repulsion、F_entropy、F_distance、F_leader、F_history 和 F_obs。", "indent": 2, "bold": False},
        {"num": 11, "text": "融合得到最终飞行方向 F_t ← w_r F_repulsion + w_e F_entropy + w_d F_distance + w_l F_leader + w_h F_history + F_obs。", "indent": 2, "bold": False},
        {"num": 12, "text": "在 AirSim 中执行动作，获取奖励 r_t、下一时刻状态 s_{t+1} 和终止标志 done。", "indent": 2, "bold": False},
        {"num": 13, "text": "将经验 (s_t, ã_t, r_t, s_{t+1}, done) 存入经验池 D，并随机采样 N 条经验。", "indent": 2, "bold": False},
        {"num": 14, "text": "计算目标值 y_i ← r_i + γQ'(s_{i+1}, μ'(s_{i+1}|θ_μ')|θ_Q')。", "indent": 2, "bold": False},
        {"num": 15, "text": "最小化损失函数更新评论家网络，并根据策略梯度更新演员网络。", "indent": 2, "bold": False},
        {"num": 16, "text": "软更新目标网络 θ_μ' 和 θ_Q'。", "indent": 2, "bold": False},
        {"num": 17, "text": "if done = True then", "indent": 2, "bold": True},
        {"num": 18, "text": "break", "indent": 3, "bold": True},
        {"num": 19, "text": "end if", "indent": 2, "bold": True},
        {"num": 20, "text": "令 s_t ← s_{t+1}。", "indent": 2, "bold": False},
        {"num": 21, "text": "end for", "indent": 1, "bold": True},
        {"num": 22, "text": "end for", "indent": 0, "bold": True},
        {"num": 23, "text": "保存仿真预训练模型 weight_predictor_airsim、best_model 和阶段一训练日志。", "indent": 0, "bold": False},
        {"num": 24, "text": "return M_sim，L_sim", "indent": 0, "bold": True},
    ],
    "caption": "图 1  双阶段实验阶段一（sim_pretrain）逻辑流程图。",
    "flowchart": STAGE1_FLOWCHART,
}

STAGE2 = {
    "title": "算法 4.4 双阶段实验的二阶段（real_weighted_refine）运行流程",
    "known": (
        "已知：阶段一输出模型：M_sim，修正模式：refine_mode ∈ {online, offline_logs}，"
        "实体机在线配置：C_online，实体机历史日志：L_real，双阶段输出目录：A_exp。"
    ),
    "goal": "求：修正后模型 M_refine，双阶段汇总结果 S_2，继续修正建议 R。",
    "lines": [
        {"num": 1, "text": "加载阶段一输出模型 M_sim；若处于继续修正回合，则加载当前阶段二模型作为新的初始模型。", "indent": 0, "bold": False},
        {"num": 2, "text": "if refine_mode = online then", "indent": 0, "bold": True},
        {"num": 3, "text": "解析默认模型路径，若存在最新 online 模型则继续使用，否则回退到阶段一仿真预训练模型。", "indent": 1, "bold": False},
        {"num": 4, "text": "连接实体 Crazyflie，构建 CrazyflieOnlineWeightEnv。", "indent": 1, "bold": False},
        {"num": 5, "text": "执行单轮或加权在线修正，在真实飞行回合结束后更新 DDPG 权重。", "indent": 1, "bold": False},
        {"num": 6, "text": "保存 online 修正模型 weight_predictor_crazyflie_online 和对应日志。", "indent": 1, "bold": False},
        {"num": 7, "text": "else if refine_mode = offline_logs then", "indent": 0, "bold": True},
        {"num": 8, "text": "读取 Crazyflie 历史飞行日志 L_real，并构建 CrazyflieLogEnv。", "indent": 1, "bold": False},
        {"num": 9, "text": "以阶段一模型 M_sim 为初始模型继续训练，利用日志完成离线修正。", "indent": 1, "bold": False},
        {"num": 10, "text": "保存 offline_logs 修正模型 weight_predictor_crazyflie_logs 和对应日志。", "indent": 1, "bold": False},
        {"num": 11, "text": "end if", "indent": 0, "bold": True},
        {"num": 12, "text": "将阶段二模型、日志和阶段元数据归档到 A_exp / real_weighted_refine / <mode>。", "indent": 0, "bold": False},
        {"num": 13, "text": "读取阶段一 sim_pretrain 与阶段二 real_weighted_refine 的训练 CSV。", "indent": 0, "bold": False},
        {"num": 14, "text": "构建 two_stage_summary.csv 和 two_stage_summary.md。", "indent": 0, "bold": False},
        {"num": 15, "text": "计算 efficiency gain 和 success gain。", "indent": 0, "bold": False},
        {"num": 16, "text": "if efficiency gain < 回退阈值 或 success gain < 回退阈值 then", "indent": 0, "bold": True},
        {"num": 17, "text": "R ← 谨慎继续修正。", "indent": 1, "bold": False},
        {"num": 18, "text": "else if efficiency gain > 显著提升阈值 或 success gain > 显著提升阈值 then", "indent": 0, "bold": True},
        {"num": 19, "text": "R ← 建议继续实飞修正。", "indent": 1, "bold": False},
        {"num": 20, "text": "else", "indent": 0, "bold": True},
        {"num": 21, "text": "R ← 当前可结束双阶段实验。", "indent": 1, "bold": False},
        {"num": 22, "text": "end if", "indent": 0, "bold": True},
        {"num": 23, "text": "return M_refine，S_2，R", "indent": 0, "bold": True},
    ],
    "caption": "图 2  双阶段实验阶段二（real_weighted_refine）逻辑流程图。",
    "flowchart": STAGE2_FLOWCHART,
}


def _run_xml(text: str, *, bold: bool = False, size: int = 24) -> str:
    props = [
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/>',
        f'<w:sz w:val="{size}"/>',
        f'<w:szCs w:val="{size}"/>',
    ]
    if bold:
        props.insert(1, "<w:b/>")
        props.insert(2, "<w:bCs/>")
    return f'<w:r><w:rPr>{"".join(props)}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def _paragraph_xml(text: str, *, bold_prefix: str | None = None, size: int = 24, center: bool = False) -> str:
    if bold_prefix and text.startswith(bold_prefix):
        prefix = bold_prefix
        remainder = text[len(prefix):]
        content = _run_xml(prefix, bold=True, size=size) + _run_xml(remainder, bold=False, size=size)
    else:
        content = _run_xml(text, bold=False, size=size)
    jc = '<w:jc w:val="center"/>' if center else ""
    return (
        "<w:p>"
        f'<w:pPr>{jc}<w:spacing w:before="0" w:after="80" w:line="360" w:lineRule="auto"/></w:pPr>'
        f"{content}"
        "</w:p>"
    )


def _title_xml(text: str) -> str:
    return (
        "<w:p>"
        "<w:pPr>"
        '<w:pBdr>'
        '<w:top w:val="single" w:sz="10" w:space="1" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="10" w:space="1" w:color="000000"/>'
        "</w:pBdr>"
        '<w:spacing w:before="80" w:after="120" w:line="360" w:lineRule="auto"/>'
        "</w:pPr>"
        f"{_run_xml(text, bold=True, size=30)}"
        "</w:p>"
    )


def _line_row_xml(number: int, text: str, *, indent: int, bold: bool) -> str:
    left_indent = indent * 520
    number_paragraph = (
        "<w:p>"
        '<w:pPr><w:jc w:val="right"/><w:spacing w:before="0" w:after="0" w:line="320" w:lineRule="auto"/></w:pPr>'
        f"{_run_xml(f'{number}:', bold=False, size=22)}"
        "</w:p>"
    )
    content_paragraph = (
        "<w:p>"
        f'<w:pPr><w:ind w:left="{left_indent}"/><w:spacing w:before="0" w:after="0" w:line="320" w:lineRule="auto"/></w:pPr>'
        f"{_run_xml(text, bold=bold, size=24)}"
        "</w:p>"
    )
    return (
        "<w:tr>"
        "<w:tc>"
        '<w:tcPr><w:tcW w:w="720" w:type="dxa"/><w:vAlign w:val="top"/></w:tcPr>'
        f"{number_paragraph}"
        "</w:tc>"
        "<w:tc>"
        '<w:tcPr><w:tcW w:w="9800" w:type="dxa"/><w:vAlign w:val="top"/></w:tcPr>'
        f"{content_paragraph}"
        "</w:tc>"
        "</w:tr>"
    )


def _table_xml(lines: list[dict]) -> str:
    rows = [_line_row_xml(line["num"], line["text"], indent=line["indent"], bold=line["bold"]) for line in lines]
    return (
        "<w:tbl>"
        "<w:tblPr>"
        '<w:tblW w:w="10520" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        '<w:tblBorders>'
        '<w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/>'
        '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/>'
        "</w:tblBorders>"
        '<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="0" w:type="dxa"/>'
        '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>'
        "</w:tblPr>"
        '<w:tblGrid><w:gridCol w:w="720"/><w:gridCol w:w="9800"/></w:tblGrid>'
        f"{''.join(rows)}"
        "</w:tbl>"
    )


def _closing_rule_xml() -> str:
    return (
        "<w:p>"
        "<w:pPr>"
        '<w:pBdr><w:bottom w:val="single" w:sz="10" w:space="1" w:color="000000"/></w:pBdr>'
        '<w:spacing w:before="60" w:after="60" w:line="200" w:lineRule="auto"/>'
        "</w:pPr>"
        "</w:p>"
    )


def _page_break_xml() -> str:
    return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"


def _image_paragraph_xml(rel_id: str, name: str, cx: int, cy: int) -> str:
    return (
        "<w:p>"
        '<w:pPr><w:jc w:val="center"/><w:spacing w:before="40" w:after="80" w:line="240" w:lineRule="auto"/></w:pPr>'
        "<w:r><w:drawing>"
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="1" name="{escape(name)}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic><pic:nvPicPr>'
        f'<pic:cNvPr id="0" name="{escape(name)}"/>'
        '<pic:cNvPicPr/></pic:nvPicPr><pic:blipFill>'
        f'<a:blip r:embed="{rel_id}"/>'
        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
        '</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
        "</w:p>"
    )


def _section_xml(section: dict, rel_id: str, cx: int, cy: int) -> list[str]:
    body = [
        _title_xml(section["title"]),
        _paragraph_xml(section["known"], bold_prefix="已知：", size=24),
        _paragraph_xml(section["goal"], bold_prefix="求：", size=24),
        _table_xml(section["lines"]),
        _closing_rule_xml(),
    ]
    if section["flowchart"].exists():
        body.extend(
            [
                _paragraph_xml(section["caption"], size=22, center=True),
                _image_paragraph_xml(rel_id, section["flowchart"].name, cx, cy),
                _closing_rule_xml(),
            ]
        )
    return body


def _document_xml() -> str:
    body: list[str] = []
    body.extend(_section_xml(STAGE1, "rId1", 4902200, 6735475))
    body.append(_page_break_xml())
    body.extend(_section_xml(STAGE2, "rId2", 5029200, 7236706))
    sect_pr = (
        "<w:sectPr>"
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1200" w:right="1000" w:bottom="1200" w:left="1000" '
        'w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:cols w:space="708"/>'
        '<w:docGrid w:linePitch="360"/>'
        "</w:sectPr>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14">'
        f"<w:body>{''.join(body)}{sect_pr}</w:body>"
        "</w:document>"
    )


def _write_docx(document_xml: str, output_path: Path) -> None:
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Default Extension="png" ContentType="image/png"/>
    <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    document_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/TwoStage_Stage1_Flowchart.png"/>
    <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/TwoStage_Stage2_Flowchart.png"/>
</Relationships>
"""
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("word/document.xml", document_xml)
        if STAGE1_FLOWCHART.exists() or STAGE2_FLOWCHART.exists():
            archive.writestr("word/_rels/document.xml.rels", document_rels_xml)
        if STAGE1_FLOWCHART.exists():
            archive.write(STAGE1_FLOWCHART, "word/media/TwoStage_Stage1_Flowchart.png")
        if STAGE2_FLOWCHART.exists():
            archive.write(STAGE2_FLOWCHART, "word/media/TwoStage_Stage2_Flowchart.png")


def _write_text_backup(path: Path) -> None:
    content: list[str] = []
    for section in (STAGE1, STAGE2):
        content.extend([section["title"], section["known"], section["goal"]])
        content.extend([f'{line["num"]}: {"    " * line["indent"]}{line["text"]}' for line in section["lines"]])
        content.extend(["", section["caption"], ""])
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_docx(_document_xml(), DOCX_PATH)
    _write_text_backup(TEXT_PATH)
    print(DOCX_PATH)
    print(TEXT_PATH)


if __name__ == "__main__":
    main()
