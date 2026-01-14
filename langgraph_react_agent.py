"""
LangGraph ReAct Agent for Security Fix Detection
使用LangChain官方的LangGraph框架（适配1.2.x）
"""
import json
from typing import Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent_tools import CodeAnalysisTools


# 定义Agent状态
class AgentState(TypedDict):
    messages: list
    cve_id: str
    commit_hash: str
    tool_calls_count: int
    is_final: bool
    result: Optional[dict]


class SecurityFixReActAgent:
    """基于LangGraph的安全修复识别Agent"""
    
    def __init__(
        self,
        repo_path: str,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_iterations: int = 5,
        verbose: bool = False
    ):
        self.code_tools = CodeAnalysisTools(repo_path)
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 初始化LLM
        llm_kwargs = {"model": model, "temperature": temperature, "api_key": api_key}
        if base_url:
            llm_kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # 当前commit hash（工具需要用到）
        self.current_commit = None
        
        # 创建工具
        self._create_tools()
        
        # 构建图
        self.app = self._build_graph()
    
    def _create_tools(self):
        """创建LangChain工具"""
        
        @tool
        def get_function_definition(function_name: str, file_path: str) -> str:
            """获取函数的完整源代码定义。
            
            ⚠️ Use ONLY if diff is incomplete and you need full function context.
            Most security fixes are clear from diff alone - don't use this by default!
            
            Args:
                function_name: 函数名称，如 'sbr_qmf_synthesis'
                file_path: 文件路径，如 'libavcodec/aacsbr.c'
            
            Returns:
                函数的完整代码
            """
            try:
                info = self.code_tools.get_function_definition(
                    function_name=function_name,
                    filepath=file_path,
                    commit=self.current_commit
                )
                if info:
                    return f"Function: {info.name}\nLocation: {file_path}:{info.start_line}-{info.end_line}\n\nCode:\n{info.body[:1500]}"
                else:
                    return f"Function '{function_name}' not found in {file_path}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        @tool
        def get_caller_context(function_name: str) -> str:
            """查找函数在代码库中的调用位置。
            
            ⚠️ RARELY NEEDED! Most security fixes don't require caller analysis.
            Static functions often have no useful callers to find.
            Only use if you specifically need to understand how the function is invoked.
            
            Args:
                function_name: 函数名称
            
            Returns:
                函数的调用位置和上下文代码（可能为空）
            """
            try:
                callers = self.code_tools.get_caller_context(
                    function_name=function_name,
                    commit=self.current_commit,
                    max_results=3
                )
                if callers:
                    result = f"Found {len(callers)} call sites:\n\n"
                    for i, c in enumerate(callers, 1):
                        result += f"[{i}] {c['file']}:{c['line']}\n{c['context']}\n\n"
                    return result
                else:
                    return f"No callers found for '{function_name}'"
            except Exception as e:
                return f"Error: {str(e)}"
        
        self.tools = [get_function_definition, get_caller_context]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
    
    def _build_graph(self):
        """构建LangGraph状态图"""
        
        def should_continue(state: AgentState):
            """判断是否继续"""
            if state["is_final"]:
                return "end"
            if state["tool_calls_count"] >= self.max_iterations:
                return "end"
            
            messages = state["messages"]
            last_message = messages[-1]
            
            # 检查是否有工具调用
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "continue"
            return "end"
        
        def call_model(state: AgentState):
            """调用LLM"""
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            
            if self.verbose:
                print(f"\n[LLM Response] {response.content[:200]}...")
            
            return {"messages": messages + [response]}
        
        def call_tools(state: AgentState):
            """调用工具"""
            messages = state["messages"]
            last_message = messages[-1]
            
            tool_outputs = []
            if hasattr(last_message, "tool_calls"):
                for tool_call in last_message.tool_calls:
                    if self.verbose:
                        print(f"[Tool Call] {tool_call['name']}({tool_call['args']})")
                    
                    # 找到对应的工具并执行
                    for tool in self.tools:
                        if tool.name == tool_call["name"]:
                            try:
                                output = tool.invoke(tool_call["args"])
                                tool_outputs.append({
                                    "role": "tool",
                                    "content": str(output),
                                    "tool_call_id": tool_call["id"]
                                })
                            except Exception as e:
                                tool_outputs.append({
                                    "role": "tool",
                                    "content": f"Error: {str(e)}",
                                    "tool_call_id": tool_call["id"]
                                })
            
            return {
                "messages": messages + tool_outputs,
                "tool_calls_count": state["tool_calls_count"] + len(tool_outputs)
            }
        
        # 构建图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tools)
        
        # 设置入口
        workflow.set_entry_point("agent")
        
        # 添加边
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def analyze_commit(
        self,
        cve_id: str,
        cve_desc: str,
        commit_hash: str,
        commit_message: str,
        repo_name: Optional[str] = None
    ) -> dict:
        """
        分析commit是否为CVE修复
        使用LangGraph管理ReAct流程
        """
        # 设置当前commit
        self.current_commit = commit_hash
        
        # 获取diff
        diff = self.code_tools.get_commit_diff(commit_hash, context_lines=8)
        if len(diff) > 4000:
            diff = diff[:4000] + "\n[... truncated ...]"
        
        # 构建system message
        system_msg = SystemMessage(content=f"""You are a security researcher analyzing Git commits to identify CVE fixes.

**Your Task:** Determine if this commit fixes CVE {cve_id}.

**Available Tools (Use SPARINGLY):**
1. get_function_definition(function_name, file_path): Get full function code
   - Use ONLY if diff shows partial function and you need complete logic
   - DO NOT use if diff already shows the key security change clearly

2. get_caller_context(function_name): Find where function is called
   - Use ONLY if you need to understand how function is used
   - Often NOT NEEDED for security fixes (most are self-contained)
   - Note: Static functions may have no callers outside their file

**Analysis Rules:**
1. **Start with the diff**: Most security fixes are obvious from diff alone
   - Boundary checks, null checks, size validations → likely fixes
   - If diff is clear, make decision WITHOUT tools

2. **CVE Specificity**: Only "YES" if commit fixes THIS SPECIFIC CVE
   - Component must match (e.g., "aacsbr.c" for AAC SBR bug)
   - Vulnerability type must match (e.g., buffer underflow vs use-after-free)

3. **Be Skeptical**: Default to "NO"
   - Refactoring/cleanup/tests/comments are NOT security fixes
   - Security fix in different component = NO

**Decision Process:**
Step 1: Check if diff shows obvious security fix → decide immediately
Step 2: If unclear, use get_function_definition ONCE if needed
Step 3: Make final decision (caller context rarely needed)

**Final Response Format:**
{{
  "decision": "YES" or "NO",
  "confidence": 0-100,
  "reasoning": "Brief explanation",
  "component_match": true/false,
  "vulnerability_match": true/false
}}""")
        
        # 构建user message
        user_msg = HumanMessage(content=f"""
Target CVE: {cve_id}
Description: {cve_desc}

Commit: {commit_hash}
Message: {commit_message}

Diff:
{diff}

Analyze if this commit fixes {cve_id}.""")
        
        # 初始状态
        initial_state = {
            "messages": [system_msg, user_msg],
            "cve_id": cve_id,
            "commit_hash": commit_hash,
            "tool_calls_count": 0,
            "is_final": False,
            "result": None
        }
        
        try:
            # 运行图
            final_state = self.app.invoke(initial_state)
            
            # 从最后的消息中提取结果
            last_message = final_state["messages"][-1]
            content = last_message.content if hasattr(last_message, "content") else str(last_message)
            
            # 尝试解析JSON
            try:
                if "{" in content and "}" in content:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    result = {
                        "decision": "NO",
                        "confidence": 0,
                        "reasoning": content[:200],
                        "component_match": False,
                        "vulnerability_match": False
                    }
            except json.JSONDecodeError:
                result = {
                    "decision": "NO",
                    "confidence": 0,
                    "reasoning": f"Parse error: {content[:200]}",
                    "component_match": False,
                    "vulnerability_match": False
                }
            
            # 构建agent_steps（为了兼容性）
            agent_steps = []
            messages = final_state["messages"]
            for i, msg in enumerate(messages):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        # 找到对应的tool输出
                        tool_output = None
                        for j in range(i+1, len(messages)):
                            if hasattr(messages[j], "get") and messages[j].get("tool_call_id") == tc["id"]:
                                tool_output = messages[j].get("content", "")
                                break
                        
                        agent_steps.append((
                            type('Action', (), {
                                'tool': tc["name"],
                                'tool_input': tc["args"]
                            })(),
                            tool_output or ""
                        ))
            
            return {
                'is_security_fix': result.get("decision", "NO").upper() == "YES",
                'confidence': result.get("confidence", 0),
                'reasoning': result.get("reasoning", ""),
                'tool_calls': final_state["tool_calls_count"],
                'agent_steps': agent_steps,
                'component_match': result.get("component_match", False),
                'vulnerability_match': result.get("vulnerability_match", False)
            }
        
        except Exception as e:
            return {
                'is_security_fix': False,
                'confidence': 0,
                'reasoning': f"Error: {str(e)}",
                'tool_calls': 0,
                'agent_steps': [],
                'component_match': False,
                'vulnerability_match': False
            }
