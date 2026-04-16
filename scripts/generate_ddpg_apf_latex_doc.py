from __future__ import annotations

import os
import zipfile
from pathlib import Path
from textwrap import dedent
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "analysis_results"
TEX_PATH = OUTPUT_DIR / "DDPG_APF_LaTeX_Algorithm.tex"
DOCX_PATH = OUTPUT_DIR / "DDPG_APF_LaTeX_Algorithm.docx"


LATEX_CONTENT = dedent(
    r"""
    \begin{algorithm}[htbp]
    \caption{DDPG+APF初始化阶段}
    \label{alg:ddpg_apf_init}
    \begin{algorithmic}[1]
    \REQUIRE 系统配置文件 $C_{sys}$，训练配置文件 $C_{train}$，奖励配置文件 $C_{reward}$，训练轮数 $E$，单轮最大步数 $T$，状态维度 $d_s=18$，动作维度 $d_a=7$
    \ENSURE 初始化后的训练系统 $M$
    \STATE 读取 $C_{sys}$、$C_{train}$ 和 $C_{reward}$
    \STATE 初始化多无人机算法服务器 $Server \leftarrow Init\_Server(C_{sys})$
    \STATE 初始化APF参数集合 $W \leftarrow \{w_r,w_e,w_d,w_l,w_h,d_o,c_o\}$
    \STATE 初始化演员网络参数 $\theta_{\mu}$ 和评论家网络参数 $\theta_Q$
    \STATE 初始化目标网络：$\theta_{\mu'} \leftarrow \theta_{\mu}, \theta_{Q'} \leftarrow \theta_Q$
    \STATE 初始化经验重放缓冲区 $D \leftarrow \emptyset$
    \STATE 初始化探索噪声过程 $\psi$
    \STATE 构建训练环境 $Env \leftarrow Init\_Env(Server, C_{train}, C_{reward})$
    \STATE 定义状态空间 $s_t \in \mathbb{R}^{18}$
    \STATE 定义动作空间 $a_t \in \mathbb{R}^{7}$
    \STATE \textbf{return} $M=\{Server, Env, \theta_{\mu}, \theta_Q, \theta_{\mu'}, \theta_{Q'}, D, \psi\}$
    \end{algorithmic}
    \end{algorithm}

    \begin{algorithm}[htbp]
    \caption{DDPG+APF学习训练阶段}
    \label{alg:ddpg_apf_train}
    \begin{algorithmic}[1]
    \REQUIRE 训练系统 $M=\{Server, Env, \theta_{\mu}, \theta_Q, \theta_{\mu'}, \theta_{Q'}, D, \psi\}$，训练轮数 $E$，单轮最大步数 $T$，批大小 $N$，折扣因子 $\gamma$，软更新系数 $\tau$
    \ENSURE 最优策略 $\mu^{*}$ 和最优APF参数集合 $W^{*}$
    \FOR{$episode = 1$ \TO $E$}
        \STATE 重置环境并获取初始状态 $s_t \leftarrow Reset(Env)$
        \STATE 重置电量、碰撞计数、出界计数和扫描状态
        \FOR{$t = 1$ \TO $T$}
            \STATE 将状态 $s_t$ 输入演员网络，得到连续动作 $a_t \leftarrow \mu(s_t \mid \theta_{\mu})$
            \STATE 加入探索噪声 $\tilde{a}_t \leftarrow a_t + \psi_t$
            \STATE 在APF安全约束下对 $\tilde{a}_t$ 进行裁剪和平滑处理
            \STATE 将 $\tilde{a}_t$ 映射为APF参数 $W_t \leftarrow \{w_r,w_e,w_d,w_l,w_h,d_o,c_o\}$
            \STATE 计算 $F_{repulsion} \leftarrow Repulsion\_Direction(s_t)$
            \STATE 计算 $F_{entropy} \leftarrow Entropy\_Direction(s_t)$
            \STATE 计算 $F_{distance} \leftarrow Distance\_Direction(s_t)$
            \STATE 计算 $F_{leader} \leftarrow Leader\_Direction(s_t)$
            \STATE 计算 $F_{history} \leftarrow History\_Direction(s_t)$
            \STATE 计算 $F_{obs} \leftarrow Obstacle\_Direction(s_t)$
            \STATE 融合APF方向分量：
            \STATE $F_t \leftarrow w_rF_{repulsion} + w_eF_{entropy} + w_dF_{distance} + w_lF_{leader} + w_hF_{history} + F_{obs}$
            \STATE 执行 $\tilde{a}_t$，并按融合方向 $F_t$ 驱动无人机飞行
            \STATE 获取奖励 $r_t$、下一时刻状态 $s_{t+1}$ 和终止标志 $done$
            \STATE 将样本 $(s_t,\tilde{a}_t,r_t,s_{t+1},done)$ 存入经验池 $D$
            \STATE 从经验池 $D$ 中采样 $N$ 条经验
            \STATE 计算目标值：
            \STATE $y_i \leftarrow r_i + \gamma Q'(s_{i+1}, \mu'(s_{i+1}\mid\theta_{\mu'}) \mid \theta_{Q'})$
            \STATE 最小化损失函数以更新评论家网络：
            \STATE $L \leftarrow \frac{1}{N}\sum_i \left[y_i - Q(s_i,a_i\mid\theta_Q)\right]^2$
            \STATE 通过策略梯度更新演员网络：
            \STATE $\nabla_{\theta_{\mu}}J \leftarrow \frac{1}{N}\sum_i \left[\nabla_a Q(s,a\mid\theta_Q)\vert_{s=s_i,a=\mu(s_i)} \cdot \nabla_{\theta_{\mu}}\mu(s\mid\theta_{\mu})\vert_{s=s_i}\right]$
            \STATE 软更新目标网络：
            \STATE $\theta_{\mu'} \leftarrow \tau\theta_{\mu} + (1-\tau)\theta_{\mu'}$
            \STATE $\theta_{Q'} \leftarrow \tau\theta_Q + (1-\tau)\theta_{Q'}$
            \STATE $s_t \leftarrow s_{t+1}$
            \IF{$done = \textbf{True}$}
                \STATE \textbf{break}
            \ENDIF
        \ENDFOR
    \ENDFOR
    \STATE \textbf{return} $\mu^{*}, W^{*}$
    \end{algorithmic}
    \end{algorithm}

    \begin{algorithm}[htbp]
    \caption{DDPG+APF输出与评测阶段}
    \label{alg:ddpg_apf_eval}
    \begin{algorithmic}[1]
    \REQUIRE 训练完成的演员网络参数 $\theta_{\mu}^{*}$，评测轮数 $E_{eval}$，单轮最大步数 $T_{eval}$，评测环境 $Env_{eval}$
    \ENSURE 评测结果集 $P$
    \STATE 加载最优演员网络参数 $\theta_{\mu}^{*}$
    \STATE 冻结网络参数
    \STATE 初始化评测结果集 $P \leftarrow \emptyset$
    \FOR{$episode = 1$ \TO $E_{eval}$}
        \STATE 重置环境并获取初始状态 $s_t$
        \FOR{$t = 1$ \TO $T_{eval}$}
            \STATE 生成动作 $a_t \leftarrow \mu(s_t \mid \theta_{\mu}^{*})$
            \STATE 将 $a_t$ 转换为APF权重和避障参数
            \STATE 按APF规则生成融合飞行方向 $F_t$
            \STATE 控制无人机群执行扫描任务
            \STATE 获取下一时刻状态 $s_{t+1}$
            \STATE 记录扫描率、平均熵、碰撞次数、电量和轨迹信息
            \IF{达到目标扫描率或触发终止条件}
                \STATE \textbf{break}
            \ENDIF
            \STATE $s_t \leftarrow s_{t+1}$
        \ENDFOR
        \STATE 将当前回合的评测结果加入 $P$
    \ENDFOR
    \STATE 导出训练日志、扫描统计结果和对比图表
    \STATE 保存最终模型、最优模型和APF参数文件
    \STATE \textbf{return} $P$
    \end{algorithmic}
    \end{algorithm}
    """
).strip()


def _paragraph_xml(text: str, *, bold: bool = False, font: str = "Consolas", size: int = 22) -> str:
    props = [
        f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="SimSun"/>',
        f'<w:sz w:val="{size}"/>',
        f'<w:szCs w:val="{size}"/>',
    ]
    if bold:
        props.insert(1, "<w:b/>")
        props.insert(2, "<w:bCs/>")
    return (
        "<w:p>"
        '<w:pPr><w:spacing w:before="0" w:after="60" w:line="320" w:lineRule="auto"/></w:pPr>'
        f'<w:r><w:rPr>{"".join(props)}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
        "</w:p>"
    )


def _title_xml(text: str) -> str:
    props = (
        '<w:pBdr>'
        '<w:top w:val="single" w:sz="12" w:space="1" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="12" w:space="1" w:color="000000"/>'
        "</w:pBdr>"
        '<w:spacing w:before="80" w:after="120" w:line="360" w:lineRule="auto"/>'
    )
    return (
        "<w:p>"
        f"<w:pPr>{props}</w:pPr>"
        '<w:r><w:rPr>'
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/>'
        '<w:b/><w:bCs/><w:sz w:val="28"/><w:szCs w:val="28"/>'
        '</w:rPr>'
        f'<w:t>{escape(text)}</w:t>'
        "</w:r>"
        "</w:p>"
    )


def _build_docx_xml() -> str:
    lines = LATEX_CONTENT.splitlines()
    body = [_title_xml("DDPG+APF LaTeX伪代码版本")]
    for line in lines:
        if line.startswith(r"\begin{algorithm}") or line.startswith(r"\end{algorithm}"):
            body.append(_paragraph_xml(line, bold=True))
        elif line.startswith(r"\caption") or line.startswith(r"\label"):
            body.append(_paragraph_xml(line, bold=True))
        else:
            body.append(_paragraph_xml(line))

    sect_pr = (
        "<w:sectPr>"
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:cols w:space="708"/>'
        '<w:docGrid w:linePitch="360"/>'
        "</w:sectPr>"
    )

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
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
    <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("word/document.xml", document_xml)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(LATEX_CONTENT + os.linesep, encoding="utf-8")
    _write_docx(_build_docx_xml(), DOCX_PATH)
    print(TEX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    main()
