import os
import json
import subprocess
from typing import Optional, Type
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool, ToolException
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from agent_tools import CodeAnalysisTools, FunctionInfo


# ==================== Pydantic Schemas ====================
class GetFunctionInput(BaseModel):
    """输入Schema for get_function_definition"""
    function_name: str = Field(..., description="函数名称，例如: sbr_qmf_synthesis")
    file_path: str = Field(..., description="文件路径，例如: libavcodec/aacsbr.c")


class GetCallerInput(BaseModel):
    """输入Schema for get_caller_context"""
    function_name: str = Field(..., description="函数名称")


# ==================== LangChain Tools ====================
class GetFunctionDefinitionTool(BaseTool):
    """工具1: 获取函数完整定义"""
    name: str = "get_function_definition"
    description: str = """
    获取指定函数的完整源代码。
    当你需要查看函数内部实现（例如：边界检查、内存操作、算法逻辑）时使用。
    
    输入参数：
    - function_name: 函数名（精确匹配）
    - file_path: 文件路径（从diff中提取）
    
    示例：get_function_definition(function_name="sbr_qmf_synthesis", file_path="libavcodec/aacsbr.c")
    """
    args_schema: Type[BaseModel] = GetFunctionInput
    
    # 自定义属性（传递给工具）
    code_tools: CodeAnalysisTools = None
    commit_hash: str = ""
    
    def _run(self, function_name: str, file_path: str) -> str:
        """同步执行"""
        if not self.code_tools:
            return "Error: CodeAnalysisTools not initialized"
        
        try:
            info = self.code_tools.get_function_definition(
                function_name=function_name,
                filepath=file_path,
                commit=self.commit_hash
            )
            
            if info:
                return f"""
Function: {info.name}
Location: {file_path}:{info.start_line}-{info.end_line}
Code:
{info.body}
"""
            else:
                return f"Function '{function_name}' not found in {file_path}"
        except Exception as e:
            raise ToolException(f"Error getting function: {e}")


class GetCallerContextTool(BaseTool):
    """工具2: 获取函数调用位置"""
    name: str = "get_caller_context"
    description: str = """
    查找函数在代码库中的调用位置（caller context）。
    当你需要了解函数如何被使用、调用者传入什么参数时使用。
    
    输入参数：
    - function_name: 函数名
    
    返回：前3个调用点及其上下文代码
    """
    args_schema: Type[BaseModel] = GetCallerInput
    
    code_tools: CodeAnalysisTools = None
    commit_hash: str = ""
    
    def _run(self, function_name: str) -> str:
        if not self.code_tools:
            return "Error: CodeAnalysisTools not initialized"
        
        try:
            callers = self.code_tools.get_caller_context(
                function_name=function_name,
                commit=self.commit_hash,
                max_results=3
            )
            
            if not callers:
                return f"No callers found for '{function_name}'"
            
            result = f"Found {len(callers)} call sites:\n\n"
            for i, c in enumerate(callers, 1):
                result += f"[{i}] {c['file']}:{c['line']}\n"
                result += f"{c['context']}\n\n"
            
            return result
        except Exception as e:
            raise ToolException(f"Error getting callers: {e}")


# ==================== ReAct Prompt Template ====================
REACT_PROMPT = """You are a senior security researcher analyzing Git commits to identify security fixes.

**Target CVE:** {cve_id}
**CVE Description:** {cve_description}

**Available Tools:**
{tools}

**Tool Names:** {tool_names}

**Critical Analysis Rules:**
1. **CVE Specificity**: Only answer "YES" if the commit fixes THIS SPECIFIC CVE
   - The modified component must match (e.g., AAC SBR decoder, not PNG decoder)
   - The vulnerability type must match (e.g., buffer underflow, not use-after-free)

2. **Use Tools Wisely** (2 tools available):
   - If diff is clear and obviously unrelated → No tools needed
   - If you see security changes but unsure about mechanism → Call get_function_definition
   - If function signature changed or need to understand usage → Call get_caller_context
   - Use tools ONLY when necessary, not by default

3. **Be Skeptical**: 
   - Default to "NO"
   - Code cleanup/refactor/tests are NOT security fixes
   - Generic security improvements in other modules are NOT fixes for THIS CVE

**Reasoning Format:**
Use this format for your thought process:

Thought: [Your reasoning about what to do next]
Action: [Tool name, or "Final Answer" if ready to conclude]
Action Input: [Tool input as JSON, or final decision]
Observation: [Tool output will appear here]
... (repeat Thought/Action/Observation as needed)

**Final Answer Format (JSON):**
{{
    "decision": "YES" or "NO",
    "confidence": 0-100,
    "reasoning": "Brief explanation of your conclusion",
    "component_match": true/false,
    "vulnerability_match": true/false,
    "tools_used": ["list", "of", "tools"]
}}

**Current Commit Analysis:**

Commit Hash: {commit_hash}
Commit Message: {commit_message}

Initial Diff Preview:
{commit_diff}

Begin your analysis!

{agent_scratchpad}
"""


# ==================== Agent Class ====================
class SecurityFixReActAgent:
    """基于LangChain ReAct的安全修复识别Agent"""
    
    def __init__(
        self,
        repo_path: str,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_iterations: int = 5,
        verbose: bool = True
    ):
        self.repo_path = repo_path
        self.code_tools = CodeAnalysisTools(repo_path)
        self.verbose = verbose
        
        # 初始化LLM
        llm_kwargs = {"model": model, "temperature": temperature, "api_key": api_key}
        if base_url:
            llm_kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Prompt模板
        self.prompt = PromptTemplate.from_template(REACT_PROMPT)
        
        # 工具列表
        self.tools = []
        self.max_iterations = max_iterations
    
    def _create_tools_for_commit(self, commit_hash: str):
        """为特定commit创建工具实例（2个核心工具）"""
        tool1 = GetFunctionDefinitionTool()
        tool1.code_tools = self.code_tools
        tool1.commit_hash = commit_hash
        
        tool2 = GetCallerContextTool()
        tool2.code_tools = self.code_tools
        tool2.commit_hash = commit_hash
        
        return [tool1, tool2]
    
    def analyze_commit(
        self,
        cve_id: str,
        cve_description: str,
        commit_hash: str,
        commit_message: str,
        repo_name: Optional[str] = None
    ) -> dict:
        """
        分析单个commit是否为CVE修复
        
        Returns:
            {
                'is_security_fix': bool,
                'confidence': int,
                'reasoning': str,
                'tool_calls': int,
                'agent_steps': list  # ReAct完整思考过程
            }
        """
        # 获取commit diff（提供足够上下文）
        diff = self.code_tools.get_commit_diff(commit_hash, context_lines=8)
        if len(diff) > 4000:
            diff = diff[:4000] + "\n\n[... Diff truncated for length ...]"
        
        # 为这个commit创建工具
        tools = self._create_tools_for_commit(commit_hash)
        
        # 创建ReAct Agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=self.prompt
        )
        
        # 创建Executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        # 执行Agent
        try:
            result = agent_executor.invoke({
                "cve_id": cve_id,
                "cve_description": cve_description,
                "commit_hash": commit_hash,
                "commit_message": commit_message,
                "commit_diff": diff
            })
            
            # 解析最终答案
            try:
                output = result.get("output", "{}")
                if isinstance(output, str):
                    final_answer = json.loads(output)
                else:
                    final_answer = output
            except json.JSONDecodeError:
                # 如果LLM没有返回JSON，尝试从文本中提取
                final_answer = {
                    "decision": "NO",
                    "confidence": 0,
                    "reasoning": output,
                    "component_match": False,
                    "vulnerability_match": False
                }
            
            # 统计工具调用次数
            intermediate_steps = result.get("intermediate_steps", [])
            tool_calls = len([s for s in intermediate_steps if s[0].tool != "Final Answer"])
            
            return {
                'is_security_fix': final_answer.get("decision", "NO").upper() == "YES",
                'confidence': final_answer.get("confidence", 0),
                'reasoning': final_answer.get("reasoning", ""),
                'tool_calls': tool_calls,
                'agent_steps': intermediate_steps,
                'component_match': final_answer.get("component_match", False),
                'vulnerability_match': final_answer.get("vulnerability_match", False)
            }
            
        except Exception as e:
            return {
                'is_security_fix': False,
                'confidence': 0,
                'reasoning': f"Agent error: {str(e)}",
                'tool_calls': 0,
                'agent_steps': [],
                'component_match': False,
                'vulnerability_match': False
            }


# ==================== 测试示例 ====================
if __name__ == "__main__":
    # 配置
    API_KEY = "your-api-key-here"
    REPO_PATH = r"E:\ptrhon project\project2\vcmatch_repro\gitrepo1\FFmpeg"
    
    # 创建Agent
    agent = SecurityFixReActAgent(
        repo_path=REPO_PATH,
        api_key=API_KEY,
        model="gpt-4o-mini",
        temperature=0.1,
        verbose=True
    )
    
    # 测试案例
    result = agent.analyze_commit(
        cve_id="CVE-2012-0850",
        cve_description="Memory corruption in FFmpeg AAC SBR decoder (sbr_qmf_synthesis function, v_off variable)",
        commit_hash="944f5b2779e4aa63f7624df6cd4de832a53db81b",
        commit_message="Fix buffer underflow in AAC SBR decoder"
    )
    
    print("\n" + "="*70)
    print("RESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
