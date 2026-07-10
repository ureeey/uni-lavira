# Claude Code CLI 完全指南

## 一、基础入门

### 安装与启动

```bash
# macOS / Linux / WSL
curl -fsSL https://claude.ai/install.sh | bash

# Homebrew
brew install --cask claude-code
```

```bash
cd your-project
claude
```

### 核心斜杠命令

| 命令 | 作用 |
|------|------|
| `/help` | 显示帮助和可用命令列表 |
| `/clear` | 开始新对话，之前内容可通过 `/resume` 恢复 |
| `/compact` | 压缩上下文释放空间（可加提示词指引压缩方向） |
| `/config` | 打开设置面板，如 `/config theme=dark` |
| `/model` | 切换 AI 模型并保存为默认 |
| `/cost` | 查看当前会话 Token 使用量和费用 |
| `/effort` | 设置推理深度：`low`/`medium`/`high`/`xhigh`/`max`/`ultracode` |
| `/fast` | 切换快速模式 |
| `/plan` | 进入计划模式（分析不修改） |
| `/memory` | 管理记忆文件、切换自动记忆 |
| `/init` | 初始化项目生成 CLAUDE.md |
| `/diff` | 可视化查看未提交变更 |
| `/rewind` | 回退到之前的检查点（代码+对话） |
| `/resume` | 恢复之前的会话 |
| `/branch` | 分支当前对话，尝试不同方向 |
| `/btw` | 快速提问，不记入对话历史 |
| `/background` / `/bg` | 分离到后台运行 |
| `/review` | 审查 GitHub PR |
| `/code-review` | 审查当前工作区 diff |
| `/simplify` | 代码简化/重构 |
| `/security-review` | 安全性审查 |
| `/doctor` | 诊断环境问题 |
| `/feedback` | 提交反馈或报 bug |

### 会话管理

- **命名会话**：`/rename session-name` 方便恢复
- **恢复会话**：`claude --resume` 或 `claude --continue`
- **分支对话**：`/branch` 不丢失原会话的前提下尝试其他方向
- **查看上下文**：`/context` 可视化展示上下文使用情况
- **导出对话**：`/export filename.txt`
- **Ctrl+C** 中断，**Ctrl+D** 退出

### 权限系统

三种权限模式：
1. **默认模式**：每步需确认
2. **自动模式**：自动批准安全操作，阻断高风险命令
3. **允许/拒绝列表**：通过 `settings.json` 配置白名单

```json
{
  "permissions": {
    "allow": ["Bash(npm run lint)", "Bash(npm test *)"],
    "deny": ["Bash(curl *)", "Read(./.env)"]
  }
}
```

### IDE 集成

- **VS Code 扩展**：侧边栏中使用，内联 diff、计划审查
- **JetBrains 插件**：IntelliJ / PyCharm / WebStorm
- **桌面应用**：多会话并行、可视 diff
- **Web 端**：claude.ai/code，无需本地环境
- **终端版**：功能最全
- 配置和设置在环境间共享

### 非交互模式

```bash
claude -p "Explain what this project does"
claude -p "List all API endpoints" --output-format json
claude -p "Analyze log" --output-format stream-json --verbose
claude -p "Fix lint errors" --allowedTools "Edit,Bash(npm run lint)"
```

---

## 二、进阶方法

### 2.1 Hooks 系统

Hooks 在生命周期特定时间点自动执行，是**确定性**的（与 CLAUDE.md 的建议性不同）。

**Hook 类型**：`command`、`http`、`mcp_tool`、`prompt`、`agent`

**关键事件**：

| 事件 | 触发时机 |
|------|----------|
| `SessionStart` / `SessionEnd` | 会话开始/结束 |
| `UserPromptSubmit` | 用户提交提示词 |
| `PreToolUse` | 工具调用前（可阻止/修改） |
| `PostToolUse` / `PostToolUseFailure` | 工具调用成功/失败后 |
| `Stop` | Claude 完成回复 |
| `PreCompact` / `PostCompact` | 上下文压缩前后 |
| `FileChanged` | 监控的文件变更 |

**示例**：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "if": "Bash(rm *)",
          "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/block-rm.sh"
        }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit",
        "hooks": [{
          "type": "command",
          "command": "npx eslint --fix ${tool_input.file_path}"
        }]
      }
    ]
  }
}
```

### 2.2 Skills（自定义斜杠命令）

Skills 是比 CLAUDE.md 更轻量的指令封装，**仅在需要时加载**。

**创建** `.claude/skills/<name>/SKILL.md`：

```markdown
---
name: deploy-staging
description: Deploy current branch to staging
disable-model-invocation: true
---

# Deploy to Staging
1. Run `npm run build` and verify
2. Run `npm run test`
3. Deploy: `npm run deploy -- --env staging`
4. Verify at https://staging.example.com
```

- `disable-model-invocation: true`：仅手动调用
- `$ARGUMENTS` 引用用户传入的参数

**CLAUDE.md vs Skill**：

| | CLAUDE.md | Skill |
|---|---|---|
| 加载时机 | 每次会话 | 仅匹配时 |
| 适用场景 | 通用指令、架构约定 | 特定领域知识、工作流 |
| 大小 | 建议 ≤200 行 | 可较大 |

### 2.3 MCP 服务器集成

```bash
claude mcp add my-server --type stdio --command "npx @modelcontextprotocol/server-github"
```

或在 `.mcp.json` / `settings.json` 中配置。支持 stdio 和 HTTP/SSE 类型。

管理命令：`/mcp`、`/mcp reconnect`、`/mcp enable/disable`

### 2.4 Settings.json 深度配置

优先级（高→低）：托管设置 > CLI 标志 > `.claude/settings.local.json` > `.claude/settings.json`（项目级）> `~/.claude/settings.json`（用户级）

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": { "allow": [], "deny": [] },
  "hooks": { "PreToolUse": [], "PostToolUse": [] },
  "env": { "MY_VAR": "value" },
  "autoMemoryEnabled": true,
  "autoCompactEnabled": true,
  "fileCheckpointingEnabled": true,
  "editorMode": "normal",
  "availableModels": ["sonnet", "haiku"],
  "fallbackModel": ["claude-sonnet-5", "claude-haiku-4-5"],
  "advisorModel": "opus"
}
```

### 2.5 CLAUDE.md — 项目指令文件

每次会话自动加载。**应该放**：构建命令、代码规范、测试指令、仓库规程、架构决策、常见陷阱。**不要放**：Claude 能自己读代码发现的、标准语言约定、频繁变化的信息。

支持 `@path` 引用外部文件，支持路径敏感规则。

### 2.6 Memory 记忆系统

- **CLAUDE.md**（手动维护）：项目根目录，每次会话加载
- **Auto Memory**（自动维护）：`.claude/projects/<hash>/memory/`，Claude 自动记录。`/memory` 管理

### 2.7 Subagents（子代理）

独立上下文窗口运行，不污染主会话。创建 `.claude/agents/<name>.md`：

```markdown
---
name: security-reviewer
description: Reviews code for security vulnerabilities
tools: Read, Grep, Glob, Bash
model: opus
---
```

**Agent Teams**：多代理自动协作，`/tasks` 查看状态。

### 2.8 Worktree 隔离

`EnterWorktree` 创建 git worktree 隔离环境，并行作业互不干扰。`/batch` 大规模并行变更。

### 2.9 后台任务 & 定时任务

- **Bash 后台**：`run_in_background: true`
- **Agent 后台**：默认后台运行
- **`/background`** / **`/fork`**：分离/派生会话到后台
- **`/loop`**：重复执行，如 `/loop 5m check deploy`
- **`/schedule`**：云端定时任务
- **`CronCreate`**：cron 式调度

### 2.10 Plan Mode（计划模式）

`/plan` 进入，可读文件、设计方案，但不能修改代码。确认后 `Esc` 退出开始实现。

### 2.11 多 Agent 工作流（Workflow）

`parallel()` 并行、`pipeline()` 流水线编排，适合大规模审查、迁移、审计。

### 2.12 键盘快捷键

| 快捷键 | 作用 |
|--------|------|
| `Enter` | 提交消息 |
| `Esc` | 中断当前操作 |
| `Ctrl+C` | 强制中断 |
| `Ctrl+D` | 退出 |
| `Ctrl+R` | 历史搜索 |
| `Ctrl+L` | 清屏（两次快速按执行 `/clear`）|
| `Ctrl+J` | 插入换行不提交 |
| `Shift+Tab` | 循环切换权限模式 |
| `Ctrl+G` | 外部编辑器打开 |
| `Ctrl+X Ctrl+K` | 停止所有后台子代理 |

在 `~/.claude/keybindings.json` 中自定义，支持 chord 组合键。

### 2.13 Vim 编辑模式

`/config editorMode=vim` 开启。

### 2.14 Statusline 状态栏

`/statusline` 配置显示内容。

---

## 三、实用技巧与最佳实践

### 上下文管理

- 不相关任务间用 `/clear` 重置
- 快速提问用 `/btw`，不占对话历史
- 调研用子代理，避免污染主会话
- 压缩时给指令：`/compact focus on the API changes`

### 验证手段（最重要）

- 提供测试用例让 Claude 自行验证
- `/goal "test suite passes"` 持续工作直到目标达成
- 测试作为 Stop hook 运行
- 子代理独立上下文复审

### 常见失败模式

| 问题 | 对策 |
|------|------|
| 一个会话堆叠多任务 | 任务间用 `/clear` |
| 多次修正同一问题 | 两次失败后 `/clear`，重写提示词 |
| CLAUDE.md 过长老被忽略 | 精简到 200 行内，不常用的移到 Skill |
| 看起来对但实际有问题 | 始终提供可验证的检查手段 |

### 其他技巧

1. 简单任务用 Haiku（快+便宜），复杂推理用 Opus/Sonnet
2. 同时启动多个 Explore agent 并行搜索，效率远高于串行
3. 能用本地脚本完成的不调用 agent，节省 token
4. 用 `fewer-permission-prompts` 自动生成权限白名单

---

## 四、典型工作流速查

```
初始化       → /init → /memory → /mcp → /permissions
任务进行中    → /plan → /model → /effort → /context → /compact → /btw
并行工作      → /batch → /tasks → /background → /fork
提交前检查    → /diff → /code-review → /security-review → /simplify
会话切换      → /clear → /resume → /branch
问题排查      → /rewind → /doctor → /feedback
```
