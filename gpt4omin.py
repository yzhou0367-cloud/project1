import pandas as pd
import subprocess
import os
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = ""  # 替换你的 Key
BASE_URL = None  # 如果有代理地址，填入，例如 "https://api.deepseek.com"

# 路径配置 (请确认路径是否正确)
REPO_ROOT = r"E:\ptrhon project\project2\vcmatch_repro\gitrepo1"
INPUT_CSV = r"E:\ptrhon project\project2\vcmatch_repro\dataset\test_cve_2012_0850_hard.csv"
OUTPUT_CSV = r"E:\ptrhon project\project2\vcmatch_repro\dataset\detailed_cot_report.csv"

# CVE 描述
CVE_DESC = "In Wireshark 3.2.0 to 3.2.6, 3.0.0 to 3.0.13, and 2.6.0 to 2.6.20, the MIME Multipart dissector could crash. This was addressed in epan/dissectors/packet-multipart.c by correcting the deallocation of invalid MIME parts."


def get_git_data(repo_path, commit):
    """获取提交内容"""
    try:
        # 限制输出长度，保留头部信息，截断过长的 diff
        cmd = ["git", "show", "--format=MSG:%s%n%b%nDIFF:", commit]
        res = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, errors='ignore', encoding='utf-8')
        output = res.stdout

        # 简单截断策略：保留前 12000 个字符
        if len(output) > 12000:
            output = output[:12000] + "\n...[Truncated Diff]"
        return output
    except Exception as e:
        print(f"Git Error ({commit}): {e}")
        return ""


def analyze_with_cot(client, git_content):
    """
    【自然语言分析】
    让模型自由输出理由，通过最后一行关键词提取结果
    """
    prompt = f"""You are a Security Auditor.
Task: Determine if the Git Commit describes the **SECURITY FIX** for the Target CVE.

[Target CVE]
{CVE_DESC}

[Git Commit]
{git_content}

[Instructions]
1. Analyze the file paths: Do they match the component mentioned in the CVE?
2. Analyze the logic: Does the code fix the specific crash/vulnerability (e.g., memory deallocation, boundary check)?
3. **Reasoning First**: Explain your findings concisely.
4. **Final Verdict**: On the very last line, strictly output "VERDICT: YES" or "VERDICT: NO".

Example Response:
The commit modifies 'packet-multipart.c' and adds a NULL check before freeing memory, which matches the CVE description about correcting deallocation.
VERDICT: YES
"""
    try:
        # 使用普通文本模式，不强制 JSON
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise security researcher."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # 低温以保证输出格式稳定
        )
        content = res.choices[0].message.content.strip()

        # === 结果解析 ===
        # 获取最后一行并转大写
        lines = content.strip().split('\n')
        last_line = lines[-1].upper()

        # 判断 YES/NO
        if "VERDICT: YES" in last_line or "VERDICT:YES" in last_line:
            decision = "YES"
        else:
            decision = "NO"  # 默认 NO，防止模型胡言乱语

        return {
            "final_decision": decision,
            "reasoning": content  # 保存整个回答作为理由
        }

    except Exception as e:
        return {"final_decision": "NO", "reasoning": f"API Error: {e}"}


def main():
    # 1. 初始化 Client
    if not BASE_URL:
        client = OpenAI(api_key=API_KEY)
    else:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 读取数据
    if not os.path.exists(INPUT_CSV):
        print(f"错误: 找不到输入文件 {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    results = []

    print(f"开始分析: {len(df)} 条数据")

    # 2. 遍历每一行
    for index, row in tqdm(df.iterrows(), total=len(df)):
        repo = row['repo']
        commit = row['commit']
        label = row['label']

        # 路径处理
        repo_path = os.path.join(REPO_ROOT, repo)

        # 特殊处理 ImageMagick 目录名不一致的情况
        if 'imagemagick' in repo.lower() and not os.path.exists(repo_path):
            candidate = os.path.join(REPO_ROOT, 'ImageMagick6')
            if os.path.exists(candidate):
                repo_path = candidate

        if not os.path.exists(repo_path):
            print(f"跳过: 仓库路径不存在 {repo_path}")
            continue

        # 获取 git 内容
        git_content = get_git_data(repo_path, commit)
        if not git_content:
            print(f"警告: 无法获取提交内容 {commit}")
            continue

        # === AI 分析 ===
        ai_result = analyze_with_cot(client, git_content)

        # 结果转换
        ai_decision_str = str(ai_result.get("final_decision"))
        ai_reasoning = ai_result.get("reasoning")

        # 将 "YES" 转为 1，其他转为 0
        ai_pred_int = 1 if ai_decision_str == "YES" else 0

        results.append({
            "commit": commit,
            "ground_truth": label,
            "ai_pred": ai_pred_int,
            "is_correct": (label == ai_pred_int),
            "reasoning": ai_reasoning  # 这里是完整的分析文本
        })

    # 3. 保存结果
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 4. 统计准确率
    correct_count = len(out_df[out_df['is_correct'] == True])
    total_count = len(out_df)

    if total_count > 0:
        acc = (correct_count / total_count) * 100
        print(f"\n[完成] 准确率: {correct_count}/{total_count} ({acc:.1f}%)")
    else:
        print("\n[完成] 没有处理任何数据。")

    print(f"结果已保存至: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
