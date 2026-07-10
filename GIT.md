# Git 操作信息

## 远程仓库

| 项目 | 值 |
|------|-----|
| Remote | `origin` |
| URL | `https://github.com/ureeey/uni-lavira.git` |
| 认证 | Personal Access Token (classic)，已嵌入 remote URL |
| 协议 | HTTPS |
| 默认分支 | `main` |
| GitHub 用户 | `ureeey` |

## 本地配置

| 项目 | 值 |
|------|-----|
| 用户名 | `ureeey` |
| 邮箱 | `15651885101@163.com` |
| GitHub CLI (`gh`) | 未安装 |

## 分支策略

当前仅 `main` 分支。开发遵循 GitHub Flow：
- `main` 为稳定分支
- 新功能/修复在独立分支上进行
- 通过 PR 合并回 `main`

## Commit 规范

执行 `git commit` 时，消息末尾添加 co-author：
```
Co-Authored-By: Claude <noreply@anthropic.com>
```

## PAT 令牌

已为此仓库配置专属 Personal Access Token (classic)：
- **Scope**: `repo`（完整仓库读写）
- **Token 已嵌入 remote URL**，`git push/pull` 无需手动认证
- 若推送失败提示 401，说明 token 已过期需重新生成

## 注意事项

- `.gitignore` 已排除 `.claude/` 等目录，CLAUDE.md、HARDWARE.md、GIT.md 等根目录文件会正常追踪。
- Token 已直接写入 remote URL，`~/.gitconfig` 的 `credential.helper` 不影响此仓库。
