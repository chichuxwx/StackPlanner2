# SP2.0 暂未完成验证项

> 更新日期：2026-07-11
> 本地实现基线：`a777d9d` + 本轮工作树修改
> 完整迁移审计与演示 Case：`docs/SP2_MIGRATION_FINAL_AUDIT_AND_DEMO_CASE.md`

## 当前结论

- SP Action Loop、三层记忆、Subagent、Artifact、HITL、SOUL、Skills 和 Run/Event 已有单元与回归覆盖。
- Gateway 可启动，`/health` 返回 200。
- `http://123.59.6.244:8000/v1` 可访问，真实 Qwen3-32B 已完成一次 SP CentralAgent smoke test。
- 当前仍未完成的是 authenticated browser E2E、真实外部工具长任务、多模态和生产跨进程恢复。
- 长期 Memory 写入默认 dry-run 是设计约束，不属于验证失败。

## 1. GitHub Lint 历史失败

[Lint Check #29086580243](https://github.com/chichuxwx/StackPlanner2/actions/runs/29086580243) 的失败点是 5 个 TSX 文件未通过 Prettier。

本轮已经格式化对应文件，并在本地通过：

- Prettier。
- ESLint。
- TypeScript。
- Frontend unit：`588 passed`。
- Next.js production build。

仍需完成：

- [ ] 推送当前修改并重新运行 GitHub Actions。
- [ ] 确认 CI 使用仓库声明的 `pnpm 10.26.2` 后仍通过。

本机 `pnpm 10.11.0` 自动下载 10.26.2 时受 npm 证书链影响；本地检查使用已安装在 `node_modules/.bin` 的锁定工具完成。

## 2. Authenticated API 与浏览器 E2E

Gateway 当前使用新的本地 SQLite 数据目录，首启时没有管理员账号。按设计：

- `/health`：200。
- `/api/*`：创建管理员前返回 401。

仍需完成：

- [ ] 由仓库使用者在 `/setup` 创建测试管理员。
- [ ] 登录后完成新建 StackPlanner thread、流式输出、Artifact 打开和 HITL resume。
- [ ] 验证桌面端与移动端关键页面。

## 3. 真实 Subagent 与外部服务

协议和 fake executor 已覆盖，以下组合仍待真实服务验证：

- [ ] Researcher 使用真实搜索服务并生成带引用的 research Artifact。
- [ ] Coder 在真实 Workspace/Sandbox 中修改文件并运行测试。
- [ ] Reporter 基于多个 Artifact 生成 report v1/v2。
- [ ] Memory recaller 从已写入的真实 DR2 Memory 召回偏好。
- [ ] Skill 白名单、用户级 custom Skill 和远程 Sandbox 组合验证。

## 4. 多模态 Perception

当前 `Qwen3-32B` 配置为 `supports_vision: false`，因此只能验证 perception 的注册、路由、上下文和 Artifact 协议，不能证明真实图片理解质量。

仍需完成：

- [ ] 配置一个 `supports_vision: true` 的模型。
- [ ] 用 PDF 页面截图、架构图和表格图片验证 perception observation。
- [ ] 检查图片正文不进入 ThreadState。

## 5. 真实长任务质量

确定性长任务回归已覆盖 recall、委派失败、reflect、replan、artifact v1/v2、HITL 和 finish。模型质量仍需实测：

- [ ] 至少运行 3 个 15 至 30 分钟 Case。
- [ ] 记录完成率、重复 Action、人工介入、token、耗时和 Artifact 质量。
- [ ] 验证 Summary/Condense/Prune 对质量与成本的影响。
- [ ] 将失败 Case 固化为可重复测试。

## 6. 持久化与生产恢复

- [ ] 使用目标生产数据库做跨进程 checkpoint restore。
- [ ] Gateway 重启后恢复 pending HITL，且不重复副作用。
- [ ] 多用户、多线程并发下验证 Artifact、Memory、SOUL 和 custom Skill 隔离。
- [ ] 在目标容器、反向代理、鉴权和日志环境中完成 smoke test。

## 7. 长期 Memory 写入边界

当前已实现：

- 自动只读 Memory snapshot。
- `RECALL_MEMORY` 主动召回并写入 TaskMemoryStack 摘要。
- CandidateExtractor、Judge 和 dry-run event。

当前刻意不做：

- 自动写入真实长期 Memory。
- 自动把单次成功流程升级为 Skill。

未来若启用，需要：

- [ ] 明确人工审批或策略门控。
- [ ] 通过 DR2 MemoryUpdater 写入，不新增 Memory Store。
- [ ] 验证去重、撤销、来源追踪、用户隔离和隐私策略。

## 已有验证证据

测试组存在重叠，数字不能直接相加。

| 范围 | 结果 |
| --- | --- |
| SP 全量 | `120 passed` |
| DR2 指定回归 | `321 passed`，1 warning |
| Frontend unit | `588 passed` |
| Prettier / ESLint / TypeScript | 通过 |
| Next production build | 通过，1 Turbopack warning |
| Gateway health | 200 |
| 真实 Qwen3-32B SP smoke | 1 Action FINISH，成功 |
