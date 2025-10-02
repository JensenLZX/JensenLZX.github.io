---
title: 'Cursor 的产品演变过程'
description: '从 Cursor 的产品发布记录，看看 Cursor 是如何打造成现在的产品的'
summary: '从 Cursor 的产品发布记录，看看 Cursor 是如何打造成现在的产品的'
date: 2025-10-02T16:04:37+08:00
lastmod: 2025-10-02T16:04:37+08:00 #更新时间
draft: true
tags: ["AI Coding", "AI", "Code Edit", "Tech"]
---
{{< lead >}}
不积跬步，无以至千里
{{< /lead >}}

略去中间的小修小补，主要看一下大的功能迭代，及其耗费的时间。

**总结：**

Cursor 早期迭代简直是在飞，三天能迭代五个版本。大功能三天上，小功能一天上俩。

到现在的 Cursor 是明显速度降下来了，但是仍然不算慢。

他们合入最新的 VSCode 可能只需要 2 天即可，真的可怕。

另一个是他们的核心功能（Agent，上下文理解，Cursor Tab）实际上做的非常非常早，积累了很多版本才到现在这一步。

**小步快跑，敏捷试错才是版本答案啊。**（我还是无法理解怎么能做这么快的。）

---

以下是基于他们产品 Change log 的迭代记录：

{{< timeline  >}}
    {{< timelineItem header="2023-03-14" badge="v0.0.37">}}
    这个是第一条 change log，这时的 Cursor 应该已经有了侧边栏 + 代码补全功能
    {{< /timelineItem >}}

    {{< timelineItem header="2023-03-18" badge="v0.1.2">}}
    内置 Terminal
    {{< /timelineItem >}}

    {{< timelineItem header="2023-03-23" badge="v0.1.5">}}
    自动 Apply 侧边栏到文本中，Lint 信息的引入，保留聊天历史
    {{< /timelineItem >}}

    {{< timelineItem header="2023-03-25" badge="v0.1.7">}}
    文件名模糊搜索功能
    {{< /timelineItem >}}

    这些迭代好快啊，不管修复的 bug，只看这些新加入的功能，迭代都异常的快。
    
    不过这个时候 Cursor 还是基于 Codemirror（一个基于 JS 的网页版补全工具）

    {{< timelineItem header="2023-03-27" badge="v0.1.9">}}
    接入 OpenAI API
    {{< /timelineItem >}}

    这个时候切 VSCode 分支了。

    {{< timelineItem header="2023-04-06" badge="v0.2.0">}}
    基于 VSCodium 开始构建
    {{< /timelineItem >}}

    {{< timelineItem header="2023-04-14" badge="v0.2.3">}}
    将鼠标 hover 在 lint 错误上，AI 可为其提供解释/修复
    {{< /timelineItem >}}

    {{< timelineItem header="2023-04-16" badge="v0.2.4">}}
    聊天界面滚动，用户可选数据存储本地
    {{< /timelineItem >}}

    {{< timelineItem header="2023-04-19" badge="v0.2.6">}}
    支持 GPT-4，通过单条 prompt 生成完整项目
    {{< /timelineItem >}}

    {{< timelineItem header="2023-04-29" badge="v0.2.8">}}
    多文件 diff，远程 SSH
    {{< /timelineItem >}}

    {{< timelineItem header="2023-05-04" badge="v0.2.9">}}
    一键导入扩展，根据整个代码库提问

    *（这么早就做了这个代码库了吗？怎么做的？）*
    {{< /timelineItem >}}

    {{< timelineItem header="2023-05-17" badge="v0.2.16">}}
    终端报错自动修复
    
    活动栏固定

    Jupyter 支持

    Diff 分块拒绝、接受
    {{< /timelineItem >}}


    {{< timelineItem header="2023-05-20" badge="v0.2.18">}}
    反馈功能
    {{< /timelineItem >}}

    {{< timelineItem header="2023-06-06" badge="v0.2.27">}}
    Codebase Context v2 升级的代码库上下文
    {{< /timelineItem >}}

    {{< timelineItem header="2023-06-24" badge="v0.2.34">}}
    Chat 升级，可以 @ 文件。
    
    粘贴代码时自动格式化
    {{< /timelineItem >}}

    {{< timelineItem header="2023-06-27" badge="v0.2.42">}}
    可以修复聊天错误

    slash 命令
    {{< /timelineItem >}}

    {{< timelineItem header="2023-07-06" badge="v0.2.44-nightly">}}
    agent 功能
    {{< /timelineItem >}}

    Cursor 做 Agent 功能好早啊，我们啥时候才开始做……

    {{< timelineItem header="2023-07-07" badge="v0.2.44">}}
    对 Python/Pylance 的支持
    {{< /timelineItem >}}

    我们现在仍然没有 Pylance 的替代品，BasePywrite 真的很难用

    {{< timelineItem header="2023-07-10" badge="v0.2.46-nightly">}}
    Interface Agent (你编写接口规范，代理为你编写测试和实现。它会确保测试通过，因此你甚至无需查看具体实现。只支持 TypeScript)
    {{< /timelineItem >}}

    {{< timelineItem header="2023-07-19" badge="v0.2.49">}}
    升级全仓库上下文
    {{< /timelineItem >}}

    {{< timelineItem header="2023-07-22" badge="v0.3.1">}}
    `Cmd + K` 支持 Jupyter
    {{< /timelineItem >}}

    {{< timelineItem header="2023-07-27" badge="v0.5.0">}}
    企业支持
    {{< /timelineItem >}}

    {{< timelineItem header="2023-07-28" badge="v0.6.0">}}
    remote-ssh 支持

    AI 代码检查
    {{< /timelineItem >}}

    {{< timelineItem header="2023-08-12" badge="v0.7.8-nightly">}}
    编辑中支持后续对话
    {{< /timelineItem >}}

    {{< timelineItem header="2023-09-05" badge="v0.9.5-nightly">}}
    Cursor-Python LSP
    {{< /timelineItem >}}

    {{< timelineItem header="2023-09-08" badge="v0.10.0">}}
    更强大的文档支持。这意味着你可以添加或移除文档，并查看其对应的 URL。

    你还能查看每个已上传文档实际被使用的网页，以及最终展示给 GPT-4 以生成答案的页面。
    {{< /timelineItem >}}

    {{< timelineItem header="2023-09-10" badge="v0.10.4">}}
    Cursor-python 对应 Pylance
    {{< /timelineItem >}}

    {{< timelineItem header="2023-09-10" badge="v0.10.4-nightly">}}
    使用 @chat 从聊天中引入上下文
    {{< /timelineItem >}}

    {{< timelineItem header="2023-09-11" badge="v0.10.6-nightly">}}
    Cursor 扩展市场
    {{< /timelineItem >}}

    {{< timelineItem header="2023-09-17" badge="v0.11.1-nightly">}}
    Cursor Python 与 Cursor Marketplace
    {{< /timelineItem >}}

    Cursor 到这个时候开始大规模操心 Python 对标 Pylance 的体验

    {{< timelineItem header="2023-09-20" badge="v0.11.2-nightly">}}
    更新代码库索引
    {{< /timelineItem >}}

    {{< timelineItem header="2023-09-22" badge="v0.11.5-nightly">}}
    稳定索引 v0.1
    {{< /timelineItem >}}

    {{< timelineItem header="2023-10-01" badge="v0.12.0">}}
    优化索引
    {{< /timelineItem >}}

    {{< timelineItem header="2023-10-20" badge="v0.13.0">}}
    基于 VS Code 1.83.1

    bash 模式
    {{< /timelineItem >}}

    {{< timelineItem header="2023-11-10" badge="v0.15.0">}}
    应用聊天建议

    Copilo++
    {{< /timelineItem >}}

    这个时候 Cursor 上线了代码改写功能！

    {{< timelineItem header="2023-11-15" badge="v0.15.6-nightly">}}
    基于 VSCode 1.84.2
    {{< /timelineItem >}}

    {{< timelineItem header="2023-11-15" badge="v0.16.0">}}
    Copilot++ 缓存

    Copilot++ 加上 lint
    {{< /timelineItem >}}

    {{< timelineItem header="2023-11-22" badge="v0.16.3-nightly">}}
    “More”>“Interpreter Mode”
    {{< /timelineItem >}}

    {{< timelineItem header="2023-11-24" badge="v0.16.4-nightly">}}
    @folder
    {{< /timelineItem >}}

    {{< timelineItem header="2023-11-30" subheader="更优的上下文聊天，更快的 Copilot++" badge="v0.18.0">}}
    1. 更优的上下文聊天：尤其是，后续追问现在更智能了！
    
    2. 更快的 Copilot++：通过多项网络优化，速度再提升数百毫秒。我们仍有数百毫秒的空间可进一步削减。
    
    3. 更可靠的 Copilot++ 变更：更少闪烁，更清晰地高亮新增内容。
    {{< /timelineItem >}}

    {{< timelineItem header="2023-12-14" badge="v0.19.0">}}
    优化 Copilot++
    {{< /timelineItem >}}

    {{< timelineItem header="2023-12-29" badge="v0.21.0">}}
    并行运行多个 Command‑K
    {{< /timelineItem >}}

    {{< timelineItem header="2023-12-30" badge="v0.21.3-nightly">}}
    基于 VS Code 1.85.1
    {{< /timelineItem >}}

    {{< timelineItem header="2024-01-03" badge="v0.21.5-nightly">}}
    按住 Command 点按 shift，执行 GPT-4 驱动的 Copilot++
    {{< /timelineItem >}}

    {{< timelineItem header="2024-01-06" badge="v0.22.0">}}
    支持 Dev Containers
    {{< /timelineItem >}}

    {{< timelineItem header="2024-01-15" badge="v0.22.1-nightly">}}
    Cursor-Fast 模型，大概参数量在 GPT-3.5 级别，效果说在 GPT 3.5-4 之间
    {{< /timelineItem >}}

    {{< timelineItem header="2024-01-18" badge="v0.23.x">}}
    code symbol 支持
    {{< /timelineItem >}}

    很明显，这个时候 Cursor 对于 LSP 的使用已经非常丰富了

    {{< timelineItem header="2024-01-25" badge="v0.24.x">}}
    @Web
    {{< /timelineItem >}}

    {{< timelineItem header="2024-02-02" badge="v0.25.x">}}
    Command-K 视觉支持
    {{< /timelineItem >}}

    {{< timelineItem header="2024-02-09" badge="v0.26">}}
    AI 预览功能：可以按住 Shfit 知道当前符号的简要说明
    {{< /timelineItem >}}

    {{< timelineItem header="2024-02-15" badge="v0.27.x">}}
    Linter 更新
    {{< /timelineItem >}}

    这一段时间 Cursor 的重心又开始关注到 LSP 上了

    {{< timelineItem header="2024-02-23" badge="v0.28.x">}}
    基于 VS Code 1.86.2
    {{< /timelineItem >}}

    {{< timelineItem header="2024-03-12">}}
    支持 Claude-3-Opus
    {{< /timelineItem >}}

    {{< timelineItem header="2024-03-20" badge="v0.30.x">}}
    更快的 Copilot++，速度提升 2 倍
    {{< /timelineItem >}}

    {{< timelineItem header="2024-04-01" badge="v0.31.x">}}
    支持更长的上下文
    {{< /timelineItem >}}

    {{< timelineItem header="2024-04-12" badge="v0.32.x">}}
    Copilot++ 体验优化
    {{< /timelineItem >}}

    这一段时间就在重点打磨 Copilot++ 了

    {{< timelineItem header="2024-05-26" badge="v0.34.x">}}
    集成 VS Code 1.89 到 Cursor
    
    全新 Cursor 预测界面（这个应该就是光标跳转功能了）
    
    Gemini 1.5 Flash 现已支持长上下文模式
    
    支持用 Copilot++ 接受局部补全
    
    提升 Copilot++ 处理 linter 错误的性能
    
    代码库搜索支持可切换的重排序器
    
    GPT-4o 的解释器模式
    {{< /timelineItem >}}

    这一条信息量很大：
    - Copilot++ 能够处理 linter 错误
    - 代码库搜索是有 reranker 的
    - 这个时候已经有了光标跳转功能

    {{< timelineItem header="2024-06-08" badge="v0.35.x">}}
    默认使用跳转了
    
    支持 ssh-tunnel 

    默认禁用 Copilot++ 的“部分接受” (看来分段使用不是一个好选择)
    {{< /timelineItem >}}

    {{< timelineItem header="2024-07-03" badge="v0.36.x">}}
    Instant Apply (这个应该就是 Cursor 让小模型进行抄写的那个 blog 描述的功能)
    {{< /timelineItem >}}

    {{< timelineItem header="2024-07-13" badge="v0.37.x">}}
    Composer 多文件编辑
    {{< /timelineItem >}}

    距离 Agent 功能 2023-07-06 推出已经一年

    {{< timelineItem header="2024-07-23" badge="v0.38.x">}}
    Copilot++ 现在支持分块流式传输（目前为测试版）

    基于 VS Code 1.91.1 （真的牛逼，我们的产品 fork 出来就定死了，更新不动了）

    默认 Claude 3.5 Sonnet
    {{< /timelineItem >}}

    这个时候开始意识到 Claude-3.5-Sonnet 其实非常强

    {{< timelineItem header="2024-08-02" badge="v0.39.x">}}
    默认分块流式的 Cursor Tab
    {{< /timelineItem >}}

    {{< timelineItem header="2024-08-22" badge="v0.40.x">}}
    全新的 Cursor Tab 模型

    TypeScript 文件提供 Cursor Tab 的 Auto Import
    {{< /timelineItem >}}

    {{< timelineItem header="2024-09-17" badge="v0.41.x">}}
    Cursor Tab 的 Auto Import 支持 Python
    {{< /timelineItem >}}

    {{< timelineItem header="2024-10-09" badge="v0.42.x">}}
    基于 VS Code 1.93.1

    Composer 历史会话
    {{< /timelineItem >}}

    {{< timelineItem header="2024-11-24" badge="v0.43.x">}}
    生成 git commit message

    在聊天/Composer 中使用 @Recommended 进行语义级上下文搜索

    缺陷定位功能
    {{< /timelineItem >}}

    {{< timelineItem header="2024-12-17" badge="v0.44.x">}}
    Agent 可查看终端退出码、可在后台运行命令，且命令现在可编辑
    
    Agent 读取 linter 错误以自动修复问题

    启用 Yolo Mode 后，Agent 可自动运行终端命令
    
    @docs、@git、@web 和 @folder 现已在 Agent 中可用
    
    Agent 会自动将更改保存到磁盘
    
    Agent 可并行决定编辑多个位置
    
    Agent 可使用更智能的应用模型重新应用编辑
    
    Composer 的更改与检查点在重载后会持久保留
    
    Cursor Tab 可一次进行更大范围的编辑
    
    更好的 Composer 更改审阅体验（UX）
    
    为 Agent 提供 4o 支持
    
    更便宜且更快的缺陷发现模型
    {{< /timelineItem >}}

    {{< timelineItem header="2025-01-23" badge="v0.45.x">}}
    cursor rules

    更强的代码库理解：我们为代码库理解训练了一个新模型。我们将在接下来的一周向 0.45 的所有用户逐步推出。
    
    Fusion 模型：我们训练了一个新的 Tab 模型，在跳转和长上下文方面有显著提升。我们也将很快向用户推出。
    {{< /timelineItem >}}

    {{< timelineItem header="2025-02-05" badge="v0.46.x">}}
    Chat Composer 合并为 Agent
    {{< /timelineItem >}}

    {{< timelineItem header="2025-03-23" badge="v0.48.x">}}
    自定义模式

    聊天标签

    更快的索引

    声音提示

    按用量计费
    {{< /timelineItem >}}

    {{< timelineItem header="2025-04-15" badge="v0.49.x">}}
    自动生成 rules
    {{< /timelineItem >}}

    {{< timelineItem header="2025-05-15" badge="v0.50.x">}}
    简化定价

    Background Agent

    Max 模式

    全新的 Tab 模型

    导出聊天

    聊天分 branch    
    {{< /timelineItem >}}

    {{< timelineItem header="2025-06-04" badge="v1.0">}}
    Bugbot

    Jupyter Notebook 中使用 Agent

    Memory
    {{< /timelineItem >}}

    {{< timelineItem header="2025-06-12" badge="v1.1">}}
    Slack
    {{< /timelineItem >}}

    {{< timelineItem header="2025-07-03" badge="v1.2">}}
    Agent 待办事项

    Message Queue

    Memories

    PR 索引与搜索

    改进的 Semantic Search Embedding

    更快的 Tab: 200ms 的 TTFT

    让 Agent 解决合并冲突

    VS Code 升级至 1.99
    {{< /timelineItem >}}

    {{< timelineItem header="2025-07-29" badge="v1.3">}}
    Agent 可以使用本机终端

    上下文用量
    {{< /timelineItem >}}

    {{< timelineItem header="2025-08-06" badge="v1.4">}}
    优化工具

    Background Agent Github 支持
    {{< /timelineItem >}}

    {{< timelineItem header="2025-08-21" badge="v1.5">}}
    Linear 中集成 Agent
    {{< /timelineItem >}}

    {{< timelineItem header="2025-09-12" badge="v1.6">}}
    自定义斜杠命令

    Summary

    MCP Resources
    {{< /timelineItem >}}

    {{< timelineItem header="2025-09-29" badge="v1.7">}}
    Agent 的自动补全

    Hooks

    Team rules

    Deeplinks

    菜单栏看 Cursor Agent

    Agent 的图像支持
    {{< /timelineItem >}}

{{< /timeline >}}