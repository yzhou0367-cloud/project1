import os
import subprocess
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from tree_sitter_languages import get_language, get_parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    print("[ERROR] tree-sitter-languages not installed. Run: pip install tree-sitter-languages")
    TREE_SITTER_AVAILABLE = False


@dataclass
class FunctionInfo:
    name: str
    start_line: int
    end_line: int
    body: str


class CodeAnalysisTools:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.parsers = {}
        self.queries = {}

        if TREE_SITTER_AVAILABLE:
            self._init_parsers()

    def _init_parsers(self):
        """初始化 Tree-sitter 解析器和查询语句"""
        try:
            for lang in ['c', 'cpp', 'python', 'php', 'java', 'javascript']:
                self.parsers[lang] = get_parser(lang)

            # SCM 查询语句：用于精准提取函数定义
            self.queries = {
                'c': get_language('c').query("""
                    (function_definition
                        declarator: (function_declarator
                            declarator: [(identifier) (field_identifier)] @name)
                    ) @func
                """),
                'cpp': get_language('cpp').query("""
                    (function_definition
                        declarator: (function_declarator
                            declarator: [(identifier) (field_identifier)] @name)
                    ) @func
                """),
                'python': get_language('python').query("""
                    (function_definition name: (identifier) @name) @func
                """),
                'php': get_language('php').query("""
                    (function_definition name: (name) @name) @func
                """)
            }
        except Exception as e:
            print(f"[WARN] Tree-sitter init warning: {e}")

    def _detect_language(self, filepath: str) -> Optional[str]:
        ext = os.path.splitext(filepath)[1].lower()
        mapping = {
            '.c': 'c', '.h': 'c',
            '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
            '.py': 'python',
            '.php': 'php',
            '.java': 'java',
            '.js': 'javascript'
        }
        return mapping.get(ext)

    def _get_file_content(self, filepath: str, commit: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ['git', 'show', f'{commit}:{filepath}'],
                cwd=self.repo_path, capture_output=True, text=True,
                encoding='utf-8', errors='replace'
            )
            return result.stdout if result.returncode == 0 else None
        except:
            return None

    def extract_functions_ast(self, filepath: str, commit: str) -> List[FunctionInfo]:
        """使用 Tree-sitter 提取函数"""
        if not self.parsers: return []
        lang = self._detect_language(filepath)
        if not lang or lang not in self.parsers: return []

        content = self._get_file_content(filepath, commit)
        if not content: return []

        try:
            parser = self.parsers[lang]
            tree = parser.parse(bytes(content, "utf8"))
            query = self.queries.get(lang)
            if not query: return []

            functions = []
            lines = content.splitlines()
            captures = query.captures(tree.root_node)

            # 简化处理：直接遍历 @func 节点
            func_nodes = [n for n, name in captures if name == 'func']

            for node in func_nodes:
                func_text = content[node.start_byte:min(node.end_byte, node.start_byte + 300)]
                func_name = "unknown"

                # 尝试从captures中获取@name
                name_nodes = [n for n, name in captures if name == 'name' and 
                             n.start_byte >= node.start_byte and n.end_byte <= node.end_byte]
                if name_nodes:
                    func_name = name_nodes[0].text.decode('utf8')
                else:
                    # 简单正则提取名字作为辅助
                    if lang in ['c', 'cpp']:
                        m = re.search(r'(\w+)\s*\(', func_text)
                        if m: func_name = m.group(1)
                    elif lang == 'php':
                        m = re.search(r'function\s+(\w+)', func_text)
                        if m: func_name = m.group(1)

                start = node.start_point[0] + 1
                end = node.end_point[0] + 1

                functions.append(FunctionInfo(
                    name=func_name, start_line=start, end_line=end,
                    body="\n".join(lines[start - 1:end])
                ))
            return functions
        except:
            return []

    def get_function_definition(self, function_name: str, filepath: str, commit: str) -> Optional[FunctionInfo]:
        # 1. 尝试 AST 解析
        funcs = self.extract_functions_ast(filepath, commit)
        for f in funcs:
            if function_name in f.name or f.name in function_name:
                return f

        # 2. 保底机制：Regex 查找并截取上下文
        content = self._get_file_content(filepath, commit)
        if content:
            lines = content.splitlines()
            for i, line in enumerate(lines):
                # 匹配 C/PHP 风格的函数定义
                if re.search(fr'\b{re.escape(function_name)}\s*\(', line):
                    start = max(0, i - 2)
                    end = min(len(lines), i + 50)  # 默认取 50 行
                    return FunctionInfo(
                        name=function_name, start_line=start + 1, end_line=end + 1,
                        body="\n".join(lines[start:end])
                    )
        return None

    def get_caller_context(self, function_name: str, commit: str, max_results: int = 3) -> List[Dict]:
        callers = []
        try:
            # git grep -n "func_name(" commit
            cmd = ['git', 'grep', '-n', f'{function_name}(', commit]
            res = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, errors='ignore')
            if res.returncode == 0:
                for line in res.stdout.splitlines()[:max_results]:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        path, lno, _ = parts[0], int(parts[1]), parts[2]
                        # 获取上下文
                        ctx_content = self._get_file_content(path, commit)
                        if ctx_content:
                            lines = ctx_content.splitlines()
                            start = max(0, lno - 3)
                            end = min(len(lines), lno + 3)
                            ctx = "\n".join(lines[start:end])
                            callers.append({'file': path, 'line': lno, 'context': ctx})
        except:
            pass
        return callers

    def get_commit_diff(self, commit: str, context_lines: int = 5) -> str:
        try:
            result = subprocess.run(
                ['git', 'show', f'--unified={context_lines}', commit],
                cwd=self.repo_path, capture_output=True, text=True, errors='ignore'
            )
            return result.stdout if result.returncode == 0 else ""
        except:
            return ""

    def analyze_diff_changes(self, diff: str) -> List[str]:
        """简单提取修改了哪些文件，供 Agent 参考"""
        return re.findall(r'^\+\+\+ b/(.+)$', diff, re.MULTILINE)