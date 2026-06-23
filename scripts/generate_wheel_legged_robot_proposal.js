const fs = require("fs");
const path = require("path");
const Module = require("module");

const bundledNodeModules =
  "/Users/yuxuan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
process.env.NODE_PATH = [process.env.NODE_PATH, bundledNodeModules]
  .filter(Boolean)
  .join(path.delimiter);
Module._initPaths();

const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  LevelFormat,
  PageBreak,
  PageNumber,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableLayoutType,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const FONT = "Microsoft YaHei";
const PAGE_WIDTH = 9360;

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
const headerShading = { fill: "1F4E78", type: ShadingType.CLEAR, color: "auto" };
const altShading = { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" };
const accentShading = { fill: "DEEAF6", type: ShadingType.CLEAR, color: "auto" };
const softBlue = { fill: "EAF3F8", type: ShadingType.CLEAR, color: "auto" };

const run = (text, opts = {}) =>
  new TextRun({
    text,
    font: FONT,
    size: opts.size || 22,
    bold: opts.bold || false,
    color: opts.color || "000000",
    italics: opts.italics || false,
  });

const p = (text, opts = {}) =>
  new Paragraph({
    children: [run(text, opts)],
    spacing: {
      before: opts.before ?? 40,
      after: opts.after ?? 80,
      line: opts.line ?? 320,
    },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    indent: opts.indent,
  });

const pMix = (runs, opts = {}) =>
  new Paragraph({
    children: runs.map((item) => run(item.t, item)),
    spacing: {
      before: opts.before ?? 40,
      after: opts.after ?? 80,
      line: opts.line ?? 320,
    },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    indent: opts.indent,
  });

const h1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [run(text, { size: 32, bold: true, color: "1F4E78" })],
    spacing: { before: 360, after: 200, line: 360 },
  });

const h2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [run(text, { size: 26, bold: true, color: "2E74B5" })],
    spacing: { before: 280, after: 160, line: 340 },
  });

const h3 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [run(text, { size: 23, bold: true, color: "1F4E78" })],
    spacing: { before: 220, after: 120, line: 320 },
  });

const bullet = (text, level = 0) =>
  new Paragraph({
    numbering: { reference: "bullets", level },
    children: [run(text)],
    spacing: { before: 30, after: 50, line: 300 },
  });

const num = (text) =>
  new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    children: [run(text)],
    spacing: { before: 30, after: 50, line: 300 },
  });

const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

const hr = () =>
  new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "2E74B5", space: 6 } },
    spacing: { before: 0, after: 200 },
  });

const cell = (text, opts = {}) => {
  const shading = opts.shading;
  const color = opts.color || (shading === headerShading ? "FFFFFF" : "000000");
  const pieces = String(text).split("\n");
  return new TableCell({
    borders,
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 100, bottom: 100, left: 140, right: 140 },
    children: pieces.map(
      (piece) =>
        new Paragraph({
          children: [
            run(piece, {
              size: opts.size || 20,
              bold: opts.bold || false,
              color,
            }),
          ],
          alignment: opts.align || AlignmentType.LEFT,
          spacing: { before: 0, after: 0, line: 290 },
        }),
    ),
  });
};

const dataTable = (headers, rows, widths, opts = {}) =>
  new Table({
    width: { size: PAGE_WIDTH, type: WidthType.DXA },
    indent: { size: 120, type: WidthType.DXA },
    columnWidths: widths,
    layout: TableLayoutType.FIXED,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((header, index) =>
          cell(header, {
            width: widths[index],
            shading: headerShading,
            bold: true,
            align: AlignmentType.CENTER,
          }),
        ),
      }),
      ...rows.map(
        (row, rowIndex) =>
          new TableRow({
            children: row.map((value, colIndex) =>
              cell(value, {
                width: widths[colIndex],
                shading: opts.noStripe ? undefined : rowIndex % 2 === 0 ? altShading : undefined,
                bold: colIndex === 0 || opts.boldFirstColumn,
                align: opts.centerColumns?.includes(colIndex) ? AlignmentType.CENTER : AlignmentType.LEFT,
              }),
            ),
          }),
      ),
    ],
  });

const noteBox = (title, body) =>
  new Table({
    width: { size: PAGE_WIDTH, type: WidthType.DXA },
    indent: { size: 120, type: WidthType.DXA },
    columnWidths: [PAGE_WIDTH],
    layout: TableLayoutType.FIXED,
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: PAGE_WIDTH, type: WidthType.DXA },
            borders: {
              top: { style: BorderStyle.SINGLE, size: 4, color: "9ECAE1" },
              bottom: { style: BorderStyle.SINGLE, size: 4, color: "9ECAE1" },
              left: { style: BorderStyle.SINGLE, size: 12, color: "1F4E78" },
              right: { style: BorderStyle.SINGLE, size: 4, color: "9ECAE1" },
            },
            shading: softBlue,
            margins: { top: 140, bottom: 140, left: 180, right: 180 },
            children: [
              pMix(
                [
                  { t: title, bold: true, color: "1F4E78" },
                  { t: body },
                ],
                { before: 0, after: 0 },
              ),
            ],
          }),
        ],
      }),
    ],
  });

const children = [];
const add = (...items) => children.push(...items);

add(
  new Paragraph({ spacing: { before: 2400 }, children: [run("")] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 200 },
    children: [run("轮足巡检机器人项目", { size: 56, bold: true, color: "1F4E78" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 100, after: 200 },
    children: [run("技 术 方 案 与 商 务 报 价 书", { size: 44, bold: true, color: "1F4E78" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 200 },
    children: [run("面向物流仓储 / 园区巡检场景", { size: 28, color: "595959" })],
  }),
  new Paragraph({ spacing: { before: 1800 }, children: [run("")] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    children: [run("项目代号：WLR-LOG-2026", { size: 24, color: "595959" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    children: [run("文档版本：V1.1（续写增强版）", { size: 24, color: "595959" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    children: [run("提交日期：2026年5月", { size: 24, color: "595959" })],
  }),
  pageBreak(),
);

add(
  h1("文档说明"),
  hr(),
  p("本方案针对甲方提出的《机器狗 / 机器人硬件底座需求》，围绕物流仓库与物流园区巡检场景，给出从硬件平台、软件算法、中台系统、项目实施、商务报价到验收运维的完整交付建议。"),
  p("方案以“成熟轮足整机平台 + 重度软件二次开发 + 数据中台 + 集群调度”为核心思路。我方不重复造硬件，而是将工程与研发资源集中投入在客户最关心、且最能形成长期壁垒的软件、算法与中台层面。"),
  p("本方案中所列技术指标、价格、工期与实施边界均为基于当前调研与初步设计的预估值，最终条款以双方正式签署的技术协议与采购合同为准。"),
  noteBox(
    "方案定位：",
    "在整机售价控制在十万元以内的前提下，交付可开箱使用、可规模部署、可持续演进至端到端智能的数据化巡检机器人产品。",
  ),
  pageBreak(),
);

add(
  h1("一、项目背景与目标"),
  hr(),
  h2("1.1 项目背景"),
  p("随着物流仓库与物流园区面积持续扩张、SKU 与货流密度持续上升，传统人工巡检在覆盖密度、成本控制和数据沉淀方面逐渐显现瓶颈。"),
  bullet("巡检盲区多：大型仓库 24 小时连续作业，夜班、高架货架区、坡道和边角区域难以做到高频人工巡检。"),
  bullet("人力成本高：园区巡检岗位招聘难、流失率高，单仓全年人工巡检支出通常处于数十万至百万元量级。"),
  bullet("数据未沉淀：人工巡检结果以纸质或粗粒度电子表单为主，不利于趋势分析、风险预测与未来模型训练。"),
  p("轮足机器人在物流仓储场景具有天然适配性。仓内地面整体平整，但存在叉车坡道、托盘边缘、电缆桥架、防撞坎等小障碍；轮足复合形态能兼顾轮式高速巡航与足式越障能力，是该场景下更均衡的工程形态。"),
  h2("1.2 项目核心目标"),
  dataTable(
    ["目标维度", "具体内容"],
    [
      ["功能完备性", "满足远程遥控、手持终端遥控、基础避障、简单跟随、预设路径自主巡检、气体检测、定点拍照、声光提示、自主回桩充电、人工快速换电等全流程。"],
      ["外形与环境", "整机外形 ≤ 1.0 m × 0.5 m × 0.6 m；防护等级 ≥ IPX3；工作温度 -10 ℃ ~ 40 ℃；移动速度 0.5 ~ 1.0 m/s；单次续航 ≥ 2 h。"],
      ["可演进性", "具备完善的 2D / 3D 环境感知能力与数据回传通道，为未来基于 VLA（Vision-Language-Action）的端到端路径规划与执行预留接口与数据基础。"],
      ["集群可扩展", "支持单仓库 / 单园区内多台机器人协同巡检，配套上位机与后台中台，支持任务分派、状态监控、地图共享、告警汇聚、运维管理。"],
      ["成本约束", "整机面向甲方出货价控制在 10 万元人民币以内，规模化部署时进一步下探，凸显“量大管饱”的成本优势。"],
    ],
    [2400, 6960],
  ),
  pageBreak(),
);

add(
  h1("二、需求理解与技术路线选择"),
  hr(),
  h2("2.1 需求要点梳理"),
  dataTable(
    ["需求类别", "关键指标", "我方理解与设计取舍"],
    [
      ["形态与外形", "轮足 / 轮式；≤1×0.5×0.6 m", "优先轮足复合形态。仓内存在叉车踏板、托盘边、防撞坎、电缆桥架等小障碍，纯轮式底盘通过性不足；纯足式速度与续航劣势明显。"],
      ["移动与导航", "0.5~1.0 m/s；遥控 + 避障 + 跟随", "巡检任务重点是稳定与安全。巡检速度上限设为 1.0 m/s，避障距离阈值与紧急停障距离按物流仓人车混行环境单独标定。"],
      ["续航与供电", "≥2 h；人工换电；自主充电", "采用双电池快拆方案，单包续航 ≥ 2 h，支持现场不停机轮换。叠加自主回桩充电，满足全天候运行。"],
      ["作业载荷", "≥2 自由度机械臂；伸展≥0.3 m", "配置 2~3 DoF 轻量化机械臂，末端通过快换接口在 5 秒内更换探头模块，兼顾通用性与维护成本。"],
      ["软件与数据", "2D / 3D 感知；集群；上位机；中台", "这是我方核心交付内容。重点投入软件与算法层面的二次开发，构建可持续演进的数据中台与调度体系。"],
    ],
    [1700, 1500, 6160],
  ),
  h2("2.2 形态路线对比与最终选型"),
  dataTable(
    ["评估维度", "纯轮式底盘", "纯四足", "轮足复合（推荐）", "综合评价"],
    [
      ["通过性", "仅适合平整地面，遇 5 cm 以上障碍易卡死", "通过性强，可跨越小障碍与楼梯", "可跨越 20 cm 级台阶与坡道，覆盖仓内绝大多数场景", "轮足占优"],
      ["巡航速度", "最高 1.5~2 m/s", "通常 0.5~0.8 m/s", "可达 3~5 m/s，巡检限速 1 m/s 余量充足", "轮足占优"],
      ["续航", "3~4 h", "1~1.5 h，关节耗电高", "2~3 h，可双电池切换", "轮足达标"],
      ["整机硬件成本", "较低", "较高", "中等，已有成熟量产平台", "可接受"],
      ["感知与算法演进", "运动维度低，难支撑 VLA", "运动复合度高但运控难度大", "感知与运动复合度适中，利于数据采集与端到端训练", "轮足占优"],
    ],
    [1560, 1950, 1950, 1950, 1950],
    { centerColumns: [4] },
  ),
  pMix([
    { t: "最终技术路线：", bold: true, color: "1F4E78" },
    { t: "以成熟量产轮足复合机器人为运动平台 + 我方自研的感知 / 应用算法层 + 自研上位机 + 自研云端数据中台 + 自研集群调度引擎。" },
  ]),
  pageBreak(),
);

add(
  h1("三、总体系统架构"),
  hr(),
  h2("3.1 系统分层架构"),
  p("系统采用“端 - 边 - 云”三层架构，兼顾物流场景内网部署、断网降级、海量数据上行以及多园区统一管理需求。"),
  h3("端：机器人本体层"),
  bullet("运动控制：使用成熟运控平台，内置标准步态控制、轮足切换、自平衡、跌倒自起立等基础能力。"),
  bullet("板载感知：3D 激光雷达、双目 / RGBD 相机、IMU、机身周向防撞条；机械臂末端搭载快换式探头。"),
  bullet("板载计算：基于 Jetson Orin 系列嵌入式 GPU 平台，部署我方感知与决策算法。"),
  h3("边：园区 / 仓库边缘服务器层"),
  bullet("部署在客户园区机房或边缘小盒子，负责单园区多机器人统一管理、地图缓存、任务分派、告警汇聚、视频流转发。"),
  bullet("内网优先：弱网或断网时核心巡检任务不中断；与云中台之间采用断点续传协议。"),
  h3("云：跨园区 / 跨客户数据中台层"),
  bullet("数据归集：巡检图像、点云、气体读数、机器人状态汇聚到统一存储，并按租户、园区、机器人三级隔离。"),
  bullet("数据治理：自动标注、数据清洗、版本管理，为后续 VLA 端到端模型训练打通数据通道。"),
  h2("3.2 关键通信链路"),
  dataTable(
    ["链路", "通信方式", "说明 / 降级方案"],
    [
      ["机器人 ↔ 手持终端", "Wi-Fi 6 / 2.4G 直连 / 蓝牙", "近场操作主用 Wi-Fi 直连；远场退化到 2.4G 遥控；蓝牙作为应急或 USB 数据导出通道。"],
      ["机器人 ↔ 边缘服务器", "园区 Wi-Fi / 4G / 5G", "室内优先 Wi-Fi；露天园区切换 4G / 5G；网络抖动时本地缓存，恢复后增量同步。"],
      ["边缘 ↔ 云中台", "HTTPS / MQTT over TLS", "低频汇总数据上行；告警与远程指令双向；支持 VPN / 专线接入。"],
      ["机器人 ↔ 充电桩", "视觉 + 红外 + 触点", "视觉粗定位 → 红外 / IMU 精对接 → 触点充电；失败自动重试三次后告警。"],
    ],
    [2000, 2400, 4960],
  ),
  pageBreak(),
);

add(
  h1("四、硬件方案"),
  hr(),
  p("我方坚持“硬件复用成熟、改装聚焦集成”原则，整机硬件由轮足平台、感知与计算扩展、机械臂与作业组件、能源与充电、外围辅助五部分组成。"),
  h2("4.1 轮足运动平台"),
  dataTable(
    ["项目", "指标"],
    [
      ["形态", "四足轮足复合，可切换轮式快速巡航与足式越障两种模式。"],
      ["站立尺寸", "约 0.8 m × 0.5 m × 0.6 m，满足甲方 ≤ 1×0.5×0.6 m 约束。"],
      ["整机重量", "约 30 kg（裸机），加装机械臂与传感器后约 33~35 kg。"],
      ["最高速度", "最高可达 3~5 m/s，巡检任务限速 1 m/s。"],
      ["越障能力", "可跨越 ≥ 20 cm 台阶、攀爬 45° 斜坡。"],
      ["电池与续航", "双电池快拆 + 热插拔，单次续航 2~3 h，支持人工 1 分钟内换电。"],
      ["防护与温度", "IP54（高于甲方 IPX3 要求），工作温度 -10 ℃ ~ 40 ℃。"],
      ["基础运控", "平台内置成熟步态控制、自平衡、路径跟随、跌倒自起立、电量自管理等底层能力，并开放 SDK 供上层调用。"],
      ["板载计算", "NVIDIA Jetson Orin 系列，算力 ≥ 70 TOPS。"],
    ],
    [3000, 6360],
  ),
  h2("4.2 感知与作业扩展"),
  bullet("机身云台高清相机：1080P / 30 fps 广角相机，支持 ±120° 水平转动、±60° 俯仰，远程图传 H.265 编码。"),
  bullet("机械臂末端快换探头：标配高清拍照模组 + 多合一气体探头，可更换为红外测温、声学探伤、二维码扫描等模块。"),
  bullet("多合一气体传感器：集成 CO、CO₂、VOC、PM2.5、可燃气体（LEL）等 5~6 路工业级探头，RS485 / Modbus 接入。"),
  bullet("声光提示组件：到达任务点后自动触发 LED 警示灯环 + 5 W 扬声器。"),
  h2("4.3 整机硬件 BOM 概览"),
  dataTable(
    ["组件", "规格说明", "数量"],
    [
      ["轮足运动平台", "成熟量产平台，含板载计算、激光雷达、深度相机、双电池、原装充电桩、运控 SDK 与保修。", "1 套"],
      ["2~3 DoF 机械臂模组", "含快换法兰、控制器、线束与安装支架。", "1 套"],
      ["多合一气体探头", "CO / CO₂ / VOC / PM2.5 / 可燃气体（LEL）等。", "1 套"],
      ["云台高清相机", "1080P / 广角 / H.265 图传。", "1 套"],
      ["声光提示模块", "可调 LED 灯环 + 5 W 扬声器。", "1 套"],
      ["手持终端", "10 英寸 Android 加固平板，IP54，内置 4G / Wi-Fi / 蓝牙。", "1 台"],
      ["装配辅件", "结构改装件、线束、固定件、防雨罩、运输箱。", "1 套"],
    ],
    [2500, 5260, 1600],
    { centerColumns: [2] },
  ),
  pageBreak(),
);

add(
  h1("五、软件与算法二次开发方案"),
  hr(),
  pMix([
    { t: "本章节是项目的核心交付内容。", bold: true, color: "C00000" },
    { t: "我方在成熟运控平台之上，构建端侧感知、应用算法、上位机、边缘服务、云端数据中台与集群调度的完整软件栈。该软件栈不仅满足当前巡检任务，也为甲方后续 VLA 端到端方向沉淀数据资产。" },
  ]),
  h2("5.1 软件整体架构"),
  dataTable(
    ["功能域", "主要模块（均为我方二次开发）"],
    [
      ["感知与决策层（端侧）", "2D / 3D 环境感知融合、动态障碍物语义分割、巡检点识别与对位、机械臂任务规划、本地任务状态机。"],
      ["运动接口适配层（端侧）", "对平台原厂 SDK 进行统一封装，提供与底层无关的运动 API，便于未来更换底盘平台时上层无感切换。"],
      ["应用任务层（端侧 + 边缘）", "巡检任务编排引擎、跟随模式、声光提示、定点拍照、气体异常实时报警、本地任务回放。"],
      ["上位机与终端层", "Web 上位机、Android 手持终端 App、远程图传播放器、急停下发。"],
      ["云端数据中台层", "机器人统一管理、集群调度引擎、任务与告警中心、巡检数据资产中台、VLA 数据集导出、OTA、运维监控。"],
    ],
    [2400, 6960],
  ),
  h2("5.2 端侧感知与决策算法"),
  h3("2D / 3D 环境感知融合"),
  bullet("3D LiDAR-SLAM：在原厂建图能力之上叠加多回环检测与稀疏地图压缩，便于跨机器人共享。"),
  bullet("RGBD 语义分割：对人员、叉车、托盘、货架、地标等仓内常见目标进行实时语义分割，输出带语义的 2D 占据栅格。"),
  bullet("3D 点云目标检测：识别人员、车辆等动态障碍并预测短时轨迹，用于安全规避。"),
  bullet("多传感器时间同步：确保图像、点云、气体读数、位姿和动作数据可同步入库。"),
  h3("巡检点识别与精确对位"),
  bullet("视觉地标识别：基于 ArUco / AprilTag / 自定义贴纸进行高精度位姿估计，目标误差 ≤ 2 cm。"),
  bullet("场景特征对位：对无人工地标点位，使用局部特征匹配复现“上一次拍照视角”。"),
  bullet("机械臂末端伺服：机身定位后，机械臂末端基于 RGBD 实时微调，确保探头 / 相机正对检测目标。"),
  h3("多模态告警与异常检测"),
  bullet("气体异常实时判定：基于滑动窗口执行阈值检测与突变检测，触发本地声光报警并上报中台。"),
  bullet("图像质量自检：自动判断模糊、曝光异常与目标遮挡，质量不合格自动重拍。"),
  bullet("跌倒 / 卡死自诊断：基于 IMU、关节力矩和视觉自监督判断异常状态，触发自起立或停机告警。"),
  h2("5.3 应用任务层与交互端"),
  bullet("巡检任务编排：将巡检任务抽象为 go_to / take_photo / extend_arm / detect_gas / play_sound / wait / report 等原子动作，并通过行为树执行。"),
  bullet("跟随模式：支持 RGBD 视觉跟随与 UWB 标签跟随，保持 1.5~2 m 安全距离。"),
  bullet("远程遥控与图传：近场 Wi-Fi 直连延迟目标 ≤ 80 ms；远场 4G / 5G + WebRTC 端到端延迟目标 ≤ 300 ms。"),
  bullet("本地数据缓存：高优先级告警即时上传，常规巡检记录延后批量上传；断网恢复后自动增量同步。"),
  h2("5.4 云端数据中台与集群调度"),
  bullet("机器人管理：注册、激活、配置下发、OTA 升级、日志收集、远程运维与故障复现包导出。"),
  bullet("集群调度：基于任务优先级、机器人位置、电量和能力标签自动派发任务，支持抢占、重派、路径协同与充电编排。"),
  bullet("数据资产中台：图像、点云、气体读数、IMU、运动指令、操作日志按统一 schema 入湖，支持版本管理与训练集导出。"),
  bullet("VLA 准备：自动记录“视觉 / 点云 → 自然语言任务指令 → 实际执行动作序列”的数据三元组，支持影子策略评估。"),
  h2("5.5 技术栈选择"),
  dataTable(
    ["分层", "技术栈"],
    [
      ["端侧算法", "Ubuntu 20.04 / ROS 2 Humble、C++17、Python 3.10、TensorRT、CUDA、OpenCV、PCL。"],
      ["应用任务层", "BehaviorTree.CPP、Nav2、自研任务编排引擎（gRPC + Protobuf）。"],
      ["上位机", "React + TypeScript + Three.js（3D 点云可视化）+ Foxglove Studio 嵌入。"],
      ["手持终端", "Android 原生（Kotlin）+ WebRTC 图传 SDK。"],
      ["云端中台", "Go + Python 微服务、Kubernetes、PostgreSQL、TimescaleDB、MinIO、Kafka、Redis。"],
      ["AI / 训练侧", "PyTorch、Hugging Face Transformers、Isaac Sim、Label Studio。"],
    ],
    [2400, 6960],
  ),
  pageBreak(),
);

add(
  h1("六、项目实施计划与交付物"),
  hr(),
  h2("6.1 总体周期"),
  p("项目从合同签订到首批量产交付，整体周期约 6 个月，分为原型期、试制期、试点期、量产期四个阶段。"),
  dataTable(
    ["阶段", "时间", "主要工作", "阶段交付物"],
    [
      ["原型期", "第 1-2 月", "需求冻结、详细方案设计、平台采购与改装、核心算法预研、中台框架搭建。", "详细技术方案 V2、原型机 1 台、中台与上位机骨架版本。"],
      ["试制期", "第 3-4 月", "感知 / 应用算法迭代、上位机与终端 App 完整化、数据中台核心模块完成、集群调度引擎首版。", "试制机 3~5 台、上位机 V1.0、中台 V1.0、用户手册初稿。"],
      ["试点期", "第 5 月", "甲方指定园区现场试点、问题清单闭环、产品化打磨。", "试点验收报告、改进 PRD、最终用户手册。"],
      ["量产期", "第 6 月起", "批量生产、出厂检验、培训、交付。", "批量整机、培训记录、运维 SOP。"],
    ],
    [1500, 1500, 3000, 3360],
  ),
  h2("6.2 最终交付清单"),
  num("整机产品：按合同约定数量交付。"),
  num("软件系统：上位机 Web 端、Android 手持终端 App、云端数据中台、机器人端镜像。"),
  num("源代码与文档：上层应用层与中台源代码（按合同约定授权方式）、端侧自研模块源代码、接口文档、运维手册、用户手册。"),
  num("培训：操作培训 2 天 × 1 期；运维培训 3 天 × 1 期；二次开发培训（可选）5 天 × 1 期。"),
  num("售后：首年免费维保；7 × 12 远程响应；现场上门 SLA 按合同约定。"),
  pageBreak(),
);

add(
  h1("七、商务报价"),
  hr(),
  pMix([
    { t: "本项目报价采用“一次性研发费（NRE）+ 量产整机单价 + 软件年度服务费”的三段式结构。", bold: true, color: "C00000" },
    { t: "该结构能保证整机出厂价控制在 10 万元以内，同时为研发投入回收与长期服务收益建立合理路径。" },
  ]),
  h2("7.1 一次性研发费（NRE）"),
  dataTable(
    ["研发模块", "主要工作量", "费用（万元）"],
    [
      ["硬件集成与结构设计", "机械臂选型 / 改造、传感器集成、结构件改装、防护提升、整机线束。", "25"],
      ["端侧感知与决策算法", "2D / 3D 感知融合、语义分割、巡检点对位、机械臂末端伺服、异常检测、VLA 数据接口。", "35"],
      ["端侧应用任务层", "任务编排引擎、跟随、声光、拍照、气体报警、本地缓存与同步。", "20"],
      ["上位机与手持终端", "Web 上位机、Android App、远程图传、远程控制、急停链路。", "25"],
      ["云端数据中台", "机器人管理、数据中台、告警、报表、OTA、运维监控、私有化部署。", "35"],
      ["集群调度引擎", "任务分派、路径协同、充电编排、异常托管。", "20"],
      ["试点联调与产品化", "现场试点、bug 收敛、用户手册、培训资料。", "15"],
      ["项目管理与质量", "PM、QA、测试、文档。", "10"],
      ["NRE 合计", "", "185"],
    ],
    [3200, 3960, 2200],
    { centerColumns: [2] },
  ),
  pMix([
    { t: "支付节点：", bold: true },
    { t: "合同签订 30% / 试制机验收 30% / 试点验收 25% / 量产首批交付 15%。" },
  ]),
  h2("7.2 量产整机单价"),
  dataTable(
    ["累计订购量", "单台出厂价", "典型配置", "适用阶段"],
    [
      ["1 ~ 9 台", "9.8 万元", "标准版", "试点 / 小批量"],
      ["10 ~ 49 台", "9.2 万元", "标准版", "初次规模采购"],
      ["50 ~ 199 台", "8.6 万元", "标准版", "园区铺开"],
      ["≥ 200 台", "≤ 7.8 万元", "标准版", "跨园区量产"],
    ],
    [2400, 2400, 2400, 2160],
    { centerColumns: [0, 1, 2, 3] },
  ),
  p("说明：以上均为含税出厂价（增值税专票 13%）；累计订购量按 12 个月滚动计算；特殊定制（如防爆版、低温版、特定气体探头）单独议价。"),
  h2("7.3 软件年度服务费"),
  dataTable(
    ["服务等级", "单台年费", "服务范围"],
    [
      ["基础版", "8,000 元", "SaaS 中台、OTA、5×8 远程支持、年度 1 次升级。"],
      ["标准版", "15,000 元", "基础版 + 7×24 远程、季度升级、24 小时备件响应。"],
      ["企业版", "28,000 元", "标准版 + 私有化部署、按需现场支持、客户定制需求池。"],
    ],
    [2700, 2700, 3960],
    { centerColumns: [0, 1] },
  ),
  h2("7.4 报价小结与商务弹性"),
  bullet("以采购 100 台为例：NRE 185 万元；整机 100 × 8.6 万 = 860 万元；首年基础服务随机赠送。"),
  bullet("第二年起：100 × 1.5 万（标准版）= 150 万元 / 年；三年期 TCO 约 1,195 万元，平均每台总持有成本约 11.95 万元 / 3 年。"),
  bullet("如甲方一次性预付不少于 300 台量产订单，整机单价可进一步谈判至 7.5 万元以内。"),
  bullet("如甲方愿意将完整数据使用权（脱敏）授予我方用于内部算法迭代，整机单价可下调 3% ~ 5%。"),
  pageBreak(),
);

add(
  h1("八、关键风险与应对"),
  hr(),
  dataTable(
    ["风险类别", "风险描述", "应对策略"],
    [
      ["硬件平台单一供应商", "底盘平台单一供应商存在涨价、断供、政策风险。", "软件架构内置运动接口适配层，上层应用与具体平台解耦；同步预研第二候选平台，可在 3 个月内完成切换。"],
      ["整机成本超出", "传感器或机械臂选型成本超预算。", "BOM 设计阶段同步评估 A / B 两套供应商方案；关键件签订年度框架协议锁定价格。"],
      ["现场环境复杂度", "不同物流仓内部布局、地标、信号环境差异大。", "试点期投入工程师驻场 2~4 周，沉淀物流仓部署 SOP，量产阶段按 SOP 实施。"],
      ["软件交付周期", "中台与集群调度复杂度高，存在延期风险。", "采用敏捷迭代 + 双周演示；中台与端侧并行开发；非核心模块排到 V2.0。"],
      ["数据安全与合规", "物流园区涉及客户敏感数据。", "中台支持完全私有化部署；图像 / 点云数据本地脱敏；提供等保 / ISO 27001 配合材料。"],
      ["人员安全", "人车混行环境下机器人与人员碰撞风险。", "三重急停 + 物理防撞条 + 速度软限位 + 工作区域虚拟围栏；首批部署完成机器人责任险投保。"],
    ],
    [1800, 3000, 4560],
  ),
  pageBreak(),
);

add(
  h1("九、关于我方与合作模式"),
  hr(),
  h2("9.1 团队背景"),
  p("项目主要由 [大学] [学院 / 实验室] 团队牵头，长期从事机器人运动控制、SLAM、机器学习方向研究，具备从算法预研到工程落地的完整能力。核心成员含教授、副教授、博士与硕士研究生，工程化团队可按甲方试点与量产节奏弹性扩展。"),
  h2("9.2 知识产权"),
  bullet("项目交付的应用层与中台软件源代码，按合同约定授予甲方使用权。"),
  bullet("项目执行过程中产出的算法专利、软件著作权，归属按双方约定。"),
  bullet("我方保留在不暴露甲方敏感数据的前提下，将通用技术能力复用至其他行业项目的权利。"),
  h2("9.3 后续合作可能"),
  bullet("VLA 端到端模型联合研发：利用本项目沉淀的数据资产，共建巡检领域大模型。"),
  bullet("新型号机器人共研：扩展至防爆、低温、户外巡检、消防侦察等衍生场景。"),
  bullet("行业标准制定：联合发布物流园区智能巡检的事实标准与白皮书。"),
  pageBreak(),
);

add(
  h1("十、验收标准与测试方案"),
  hr(),
  p("为避免“能演示但不可交付”的工程风险，本项目建议采用“实验室出厂验收 + 现场试点验收 + 批量抽检验收”的三级验收机制。"),
  h2("10.1 核心验收指标"),
  dataTable(
    ["验收项", "验收标准", "测试方法"],
    [
      ["整机尺寸与防护", "外形尺寸满足 ≤ 1.0 m × 0.5 m × 0.6 m；防护等级不低于 IPX3，目标 IP54。", "尺寸实测；防泼溅测试；外观与结构检查。"],
      ["移动能力", "常规巡检速度 0.5~1.0 m/s；可通过 20 cm 级小台阶或等效障碍；坡道通行稳定。", "标准测试场地往返 10 次，统计通行成功率与异常停机次数。"],
      ["续航与换电", "单次续航 ≥ 2 h；人工换电时间 ≤ 1 min；低电量回桩策略正常触发。", "连续巡航测试；换电计时；低电量模拟。"],
      ["自主巡检", "按预设路径完成巡检点到达、定点拍照、气体检测与任务上报；点位漏检率 ≤ 2%。", "现场设置不少于 20 个巡检点，连续执行 5 轮。"],
      ["避障与安全", "静态障碍绕行；动态人员靠近时减速 / 停障；急停链路触发时间 ≤ 300 ms。", "人员横穿、叉车模拟、软硬件急停联动测试。"],
      ["气体检测", "各路传感器读数稳定上传；超阈值告警端侧与中台均可触发。", "标准气体或模拟信号注入，核对上报数值与告警链路。"],
      ["中台与集群", "机器人状态、电量、位置、任务、告警可实时展示；不少于 5 台机器人并发调度无死锁。", "边缘服务器部署，多机压力测试与异常重派测试。"],
      ["数据资产", "图像、点云、气体、位姿、动作日志按统一 schema 入库；可导出 ROS bag / Parquet / WebDataset。", "抽样核对字段完整性、时间戳对齐与导出文件可读性。"],
    ],
    [1900, 4160, 3300],
  ),
  h2("10.2 现场试点测试流程"),
  num("试点准备：完成场地踏勘、Wi-Fi / 5G 覆盖评估、充电桩点位确认、巡检路线与巡检点清单冻结。"),
  num("地图构建：机器人完成全场建图，输出 2D 栅格图、3D 点云地图与关键地标清单。"),
  num("任务配置：在上位机配置巡检剧本，包含巡检点、动作、阈值、告警接收人和数据留存策略。"),
  num("连续运行：连续运行不少于 7 天，覆盖白班、夜班、换班、弱网、充电与人工换电等典型情况。"),
  num("问题闭环：对试点问题按 P0 / P1 / P2 分级，P0 必须试点期内关闭，P1 进入量产前关闭，P2 纳入版本计划。"),
  h2("10.3 验收文件清单"),
  bullet("出厂测试报告、现场部署报告、试点运行日报 / 周报、问题清单与闭环记录。"),
  bullet("用户手册、运维手册、接口文档、软件版本清单、设备资产清单。"),
  bullet("培训签到表、培训材料、验收确认单、备件与保修说明。"),
  pageBreak(),
);

add(
  h1("十一、部署、运维与服务保障"),
  hr(),
  h2("11.1 部署条件"),
  dataTable(
    ["类别", "建议条件", "说明"],
    [
      ["网络", "仓内 Wi-Fi 覆盖 RSSI ≥ -65 dBm；关键区域建议冗余 AP；室外园区可补 4G / 5G。", "弱网区域可运行本地缓存与断点续传，但实时图传质量会下降。"],
      ["充电", "每 5~10 台机器人配置 1 个充电桩；充电桩附近预留 1.5 m × 1.5 m 无障碍区域。", "具体比例取决于任务密度、续航余量与夜间巡检频次。"],
      ["服务器", "试点期可使用单台边缘服务器；规模部署建议边缘服务器 + 云中台双层架构。", "私有化部署需提前确认机房、电源、网络与安全策略。"],
      ["场地", "巡检路线应避开长期堆放区、强反光区域和极端粉尘 / 水汽区域。", "确需通过复杂区域时，需单独进行地标、限速与安全策略标定。"],
      ["人员", "每园区建议配置 1 名设备管理员、若干现场操作员和 1 名 IT 对接人。", "我方提供培训与运维 SOP，降低日常使用门槛。"],
    ],
    [1800, 4300, 3260],
  ),
  h2("11.2 运维 SOP"),
  bullet("日检：检查电池电量、轮足关节外观、传感器镜头、急停按钮、充电触点和网络连接。"),
  bullet("周检：检查机械臂快换接口、线束松动、传感器标定状态、地图漂移情况和告警闭环记录。"),
  bullet("月检：执行软件版本更新、日志归档、备份检查、关键备件盘点与典型任务回归测试。"),
  bullet("故障处理：机器人异常停机后优先远程诊断；无法恢复时按 SOP 进行安全搬运、断电、换电或备件替换。"),
  h2("11.3 服务响应"),
  dataTable(
    ["故障级别", "定义", "响应与恢复目标"],
    [
      ["P0", "涉及人员安全、批量设备不可用、关键业务中断。", "30 分钟远程响应，4 小时内给出恢复方案；必要时安排现场支持。"],
      ["P1", "单台或少量设备无法完成核心巡检任务。", "2 小时远程响应，24 小时内给出修复或替代方案。"],
      ["P2", "非核心功能异常、报表或界面体验问题。", "1 个工作日响应，纳入迭代版本修复。"],
      ["P3", "咨询、配置、培训与优化建议。", "按服务等级排期处理。"],
    ],
    [1600, 3600, 4160],
  ),
  pageBreak(),
);

add(
  h1("十二、边界条件与附录"),
  hr(),
  h2("12.1 默认边界条件"),
  bullet("报价默认不包含防爆认证、极寒 / 高温特种改造、危险化学品场景专项认证、客户专用公网流量费与第三方系统深度集成费用。"),
  bullet("软件私有化部署默认以单园区为单位，跨园区高可用、灾备、多活架构需单独评估。"),
  bullet("若甲方要求采购指定品牌底盘、指定气体探头或指定云平台，硬件 BOM 与实施费用需重新核算。"),
  bullet("若现场存在强粉尘、强水汽、强磁、爆炸性气体、无遮挡强逆光等特殊工况，需在试点前开展专项风险评估。"),
  h2("12.2 变更管理"),
  p("项目执行中如出现需求范围、验收指标、试点场地、硬件品牌、部署架构或交付数量变化，双方应通过变更单确认影响范围、费用、工期与责任边界。未经双方书面确认的变更不进入当前里程碑验收范围。"),
  h2("12.3 版本记录"),
  dataTable(
    ["版本", "日期", "说明"],
    [
      ["V1.0", "2026年5月", "Claude 生成的初始技术方案与商务报价书。"],
      ["V1.1", "2026年5月", "补充验收标准、部署运维、服务保障与边界条件，并生成可交付 DOCX。"],
    ],
    [1800, 2200, 5360],
  ),
  new Paragraph({ spacing: { before: 600 }, children: [run("")] }),
  hr(),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 80 },
    children: [run("本文档至此结束", { size: 22, color: "595959", italics: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 80 },
    children: [run("感谢甲方对本项目的关注，期待与您建立长期合作。", { size: 22, color: "595959" })],
  }),
);

const doc = new Document({
  creator: "Project Team",
  title: "轮足巡检机器人项目技术方案与商务报价书",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: "1F4E78" },
        paragraph: { spacing: { before: 360, after: 200, line: 360 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: "2E74B5" },
        paragraph: { spacing: { before: 280, after: 160, line: 340 }, outlineLevel: 1 },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: "1F4E78" },
        paragraph: { spacing: { before: 220, after: 120, line: 320 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "●",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
          {
            level: 1,
            format: LevelFormat.BULLET,
            text: "○",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1440, hanging: 360 } } },
          },
        ],
      },
      {
        reference: "numbers",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              alignment: AlignmentType.RIGHT,
              children: [
                run("轮足巡检机器人项目 · 技术方案与商务报价书", {
                  size: 18,
                  color: "808080",
                }),
              ],
              border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF", space: 4 } },
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                run("第 ", { size: 18, color: "808080" }),
                new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "808080" }),
                run(" 页 / 共 ", { size: 18, color: "808080" }),
                new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 18, color: "808080" }),
                run(" 页", { size: 18, color: "808080" }),
              ],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

async function main() {
  const outPath =
    process.argv[2] ||
    path.resolve(process.cwd(), "outputs", "轮足巡检机器人项目-技术方案与报价书.docx");
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(outPath, buffer);
  console.log(`OK: ${outPath} size=${buffer.length} bytes`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
