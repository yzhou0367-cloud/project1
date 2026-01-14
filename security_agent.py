import os
import json
from typing import Dict, List
from openai import OpenAI
from agent_tools import CodeAnalysisTools

class SecurityFixAgent:
    def __init__(self, repo_path: str, api_key: str, base_url: str = None,
                 model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.repo_path = repo_path
        self.tools = CodeAnalysisTools(repo_path)
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_llm(self, messages: List[Dict]) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"decision": "NO", "reasoning": f"LLM Error: {e}", "confidence": 0}

    def analyze_commit(self, cve_id: str, cve_desc: str, 
                      commit_hash: str, commit_msg: str, 
                      repo_name: str) -> Dict:
        
        # 1. 动态路径修正 (ImageMagick)
        actual_path = os.path.join(self.repo_path, repo_name)
        if 'imagemagick' in repo_name.lower() and not os.path.exists(actual_path):
             actual_path = os.path.join(self.repo_path, 'ImageMagick6')
        self.tools.repo_path = actual_path

        # 2. 获取基础信息
        diff = self.tools.get_commit_diff(commit_hash, context_lines=5)
        if not diff:
            return {"is_security_fix": False, "confidence": 0, "reasoning": "No diff found", "tool_calls": 0}
        
        modified_files = self.tools.analyze_diff_changes(diff)

        # 3. 构建 Prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(cve_id, cve_desc, commit_msg, diff, modified_files)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        # 4. Round 1 推理
        print(f"\n{'='*70}")
        print(f"[ANALYZE] {cve_id} - {commit_hash[:8]}")
        print(f"{'='*70}")
        print(f"[Round 1] Initial Analysis...")
        r1 = self._call_llm(messages)
        print(f"\n[Round 1] LLM Response:")
        print(f"  Decision: {r1.get('decision', 'UNKNOWN')}")
        print(f"  Confidence: {r1.get('confidence', 0)}%")
        print(f"  Need More Info: {r1.get('need_more_info', False)}")
        print(f"\n  Reasoning (Round 1):")
        print(f"  {'-'*66}")
        reasoning1 = r1.get('reasoning', 'N/A')
        for line in reasoning1.split('\n'):
            print(f"  {line}")
        print(f"  {'-'*66}")
        messages.append({"role": "assistant", "content": json.dumps(r1)})
        
        tool_calls = 0

        # 5. Round 2 工具调用 (如果需要)
        if r1.get("need_more_info"):
            tool_req = r1.get("tool_request", {})
            tool_name = tool_req.get("type")
            target_func = tool_req.get("function_name")
            target_file = tool_req.get("file_path")

            tool_output = None
            if tool_name == "get_function_code" and target_func and target_file:
                print(f"  [Tool] Calling: get_function_definition('{target_func}', '{target_file}')")
                info = self.tools.get_function_definition(target_func, target_file, commit_hash)
                if info:
                    tool_output = info.body[:2000]
                    print(f"  [Tool] ✓ Found function (lines {info.start_line}-{info.end_line})")
                    print(f"  [Tool] Code preview:\n{tool_output[:200]}...")
                else:
                    tool_output = f"Function '{target_func}' not found."
                    print(f"  [Tool] ✗ Function not found")
                tool_calls += 1
            
            elif tool_name == "get_caller_context" and target_func:
                print(f"  [Tool] Calling: get_caller_context('{target_func}')")
                callers = self.tools.get_caller_context(target_func, commit_hash)
                if callers:
                    tool_output = "\n".join([f"{c['file']}:{c['line']}\n{c['context']}" for c in callers])
                    print(f"  [Tool] ✓ Found {len(callers)} call sites")
                    for i, c in enumerate(callers[:3], 1):
                        print(f"  [Tool]   {i}. {c['file']}:{c['line']}")
                else:
                    tool_output = "No callers found."
                    print(f"  [Tool] ✗ No callers found")
                tool_calls += 1

            if tool_output:
                messages.append({"role": "user", "content": f"Tool Output:\n{tool_output}\n\nMake FINAL decision."})
                print(f"\n[Round 2] Deep Analysis with Tool Output...")
                r_final = self._call_llm(messages)
                print(f"\n[Round 2] FINAL LLM Response:")
                print(f"  Decision: {r_final.get('decision', 'UNKNOWN')}")
                print(f"  Confidence: {r_final.get('confidence', 0)}%")
                print(f"\n  Reasoning (Round 2 - FINAL):")
                print(f"  {'-'*66}")
                reasoning2 = r_final.get('reasoning', 'N/A')
                for line in reasoning2.split('\n'):
                    print(f"  {line}")
                print(f"  {'-'*66}")
            else:
                print(f"  [Warning] Tool returned no output, using Round 1 decision")
                r_final = r1 # 工具失败，维持原判
        else:
            print(f"\n[Round 1] Sufficient information, no tools needed")
            r_final = r1

        # 6. 结果格式化
        is_fix = str(r_final.get("decision", "NO")).upper() == "YES"
        
        print(f"\n{'='*70}")
        print(f"[FINAL RESULT]")
        print(f"  Classification: {'✓ SECURITY FIX' if is_fix else '✗ NOT A FIX'}")
        print(f"  Confidence: {r_final.get('confidence', 0)}%")
        print(f"  Tool Calls: {tool_calls}")
        print(f"{'='*70}\n")
        
        return {
            'is_security_fix': is_fix,
            'confidence': r_final.get("confidence", 0),
            'reasoning': r_final.get("reasoning", ""),
            'tool_calls': tool_calls
        }

    def _build_system_prompt(self):
        return """You are a senior security researcher. Verify if a Git Commit fixes a **SPECIFIC CVE**.

**AVAILABLE TOOLS:**
1. `get_function_code(file_path, function_name)`: Get full implementation to check internal logic (e.g., boundary checks).
2. `get_caller_context(file_path, function_name)`: Check how a function is called elsewhere (e.g., if signature changed).

**CRITICAL RULES:**
1. **CVE Specificity**: "YES" ONLY if the commit fixes **THIS SPECIFIC CVE**, not just "any security issue".
   - The vulnerability type MUST match (e.g., buffer overflow vs use-after-free).
   - The affected component MUST match (e.g., AAC SBR decoder vs H.264 parser).
   - A generic security improvement in a different module is NOT a fix for THIS CVE.

2. **Be Skeptical**: Default to "NO". Only "YES" if you see clear security evidence (sanitization, bound checks, safe functions) **AND** it matches the CVE description.

3. **Refactors are NO**: Code cleanup, renaming, or moving code is NOT a security fix.

4. **Tests are NO**: Adding test cases is NOT a fix.

5. **Output JSON** (ALWAYS):
{
    "decision": "YES/NO/UNCERTAIN",
    "confidence": 0-100,
    "reasoning": "...",
    "need_more_info": true/false,
    "tool_request": {
        "type": "get_function_code" OR "get_caller_context",
        "file_path": "path/to/file.c",  // REQUIRED: Extract from diff (e.g., "libavcodec/aacsbr.c")
        "function_name": "exact_function_name"
    }
}

**IMPORTANT**: If need_more_info=true, you MUST provide complete tool_request with both file_path and function_name."""

    def _build_user_prompt(self, cve, desc, msg, diff, files):
        return f"""
CVE: {cve}
CVE Description: {desc}
Commit Message: {msg}
Modified Files: {json.dumps(files)}

Diff:
{diff[:3000]}

**Your Task:**
Does this commit fix **{cve}** specifically?

**Verification Checklist:**
1. Does the modified code relate to the component mentioned in CVE description?
   (e.g., if CVE mentions "AAC SBR decoder", changes must be in aacsbr.c or related files)

2. Does the change address the vulnerability type described in the CVE?
   (e.g., if CVE mentions "buffer overflow", look for boundary checks/buffer size fixes)

3. REJECT if:
   - Security fix but in unrelated component (e.g., PNG decoder when CVE is about AAC)
   - Generic hardening that doesn't target THIS vulnerability
   - Changes in test files, documentation, or unrelated modules

4. If diff is truncated/unclear about the fix mechanism, request tool to see full function.
"""