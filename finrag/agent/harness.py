# -*- coding: utf-8 -*-
"""
harness.py — 自研轻量 Agent 运行时（harness）

把原本散落在 agent.py / orchestrator.py 的 Agent 运行时原语，收敛成一个清晰的【门面层】，
让"我的 Agent harness 长什么样"一个文件就能讲清。它【委托】给已验证的内核，不重写逻辑——
所以是零风险的封装，不影响现有 demo。

运行时分层（每层都指到真实实现）：
  ① 工具层     ：注册表(schema) + 分派表(impl)               → agent.TOOLS / agent.TOOLS_IMPL
  ② 调度循环   ：function-calling 多轮 + max_turns 护栏        → agent._loop
  ③ 可靠性     ：LLM 调用指数退避重试 + 子进程隔离/超时         → agent.chat / robust_parse
  ④ 上下文/记忆：多轮 history + 截断防 token 膨胀               → agent.run(history=)
  ⑤ 自我修正   ：Reflexion 式自检→不足则补调工具重答(可选)       → agent.reflect_check
  ⑥ 可观测/成本：LLM 调用数 / token / 工具调用数                → agent.RUN_STATS
  ⑦ 多 Agent 编排：Planner 拆解 → 并行 Workers → Aggregator     → orchestrator.brief
  ⑧ 护栏       ："幻觉锁在工具内" + 职责边界/拒答(system)         → agent.SYSTEM

注意定位：这是【领域专用 Agent 运行时】，非 LangChain/Claude-Code 那种带沙箱、上下文压缩、
权限系统的【通用 harness 框架】——清楚这个边界，别越界吹。

用法：
  from finrag.agent.harness import AgentHarness
  h = AgentHarness()
  answer, trace = h.run("对比宁德和隆基2026盈利分歧", reflect=True)
  print(h.stats)                                  # 本次调用的成本/可观测指标
  brief = h.run_parallel("出一份宁德多维投研简报")    # 多 Agent 并行编排
  python harness.py "你的问题"                      # 命令行直接体验
"""
import finrag.agent.agent as agent
import finrag.agent.orchestrator as orchestrator
class AgentHarness:
    """领域专用 Agent 运行时门面：包装【真实】运行时原语，便于演示与复用。"""

    def __init__(self):
        # 直接引用真实注册表/分派表（不是副本）——harness 暴露的就是运行时本体
        self.tools = agent.TOOLS
        self.impls = agent.TOOLS_IMPL
        self.system = agent.SYSTEM

    # ① 工具层 -----------------------------------------------------------
    def register_tool(self, schema, impl):
        """注册一个工具：schema(function-calling 描述) + impl(可调用)。返回工具名。"""
        name = schema["function"]["name"]
        self.tools.append(schema)
        self.impls[name] = impl
        return name

    def list_tools(self):
        return [t["function"]["name"] for t in self.tools]

    # ②~⑥ 单 Agent 运行（调度循环 + 重试 + 上下文 + 自我修正 + 可观测）-----
    def run(self, question, history=None, reflect=False, tools=None, system=None, verbose=False):
        """跑一次 Agent。返回 (answer, trace)；本次成本/指标见 .stats。"""
        return agent.run(question, history=history, reflect=reflect,
                         tools=tools, system=system, verbose=verbose)

    # ⑦ 多 Agent 编排 ----------------------------------------------------
    def run_parallel(self, task, parallel=True, verbose=False):
        """Planner→并行 Workers→Aggregator 多 Agent 编排，产结构化投研简报。"""
        return orchestrator.brief(task, parallel=parallel, verbose=verbose)

    # ⑥ 可观测 ----------------------------------------------------------
    @property
    def stats(self):
        """最近一次 run 的成本/可观测指标：LLM 调用数 / token / 工具调用数 / 是否自检。"""
        return dict(agent.RUN_STATS)


# 进程内默认单例（供 app.py / api.py 直接复用）
default = AgentHarness()


if __name__ == "__main__":
    import sys
    h = AgentHarness()
    print("Agent 运行时已装载，工具：", h.list_tools())
    q = " ".join(sys.argv[1:]) or "对比宁德时代和隆基绿能 2026 年归母净利增速，谁的卖方分歧更大？"
    print(f"\n❓ {q}\n")
    ans, trace = h.run(q, reflect=False)
    print(ans)
    print("\n📊 运行时指标：", h.stats)
