import pandas as pd
import subprocess
import os
import json
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = "sk-proj-4NpyZOAR9q6IKfliLwoK2vr_OygnJ8w8VSqFb2A_Ni8XWxs7wCpUIJUkIDdm0I4-rP56266up9T3BlbkFJcYReKgOACl_hsM5zRfScy2ybT0nNVa_kt_9aRZnEsG6LLFCuZzIjXiEMfyZ7Gzwc9krPmixsNAA"  # 替换你的 Key
BASE_URL = None # 或 None

# 路径配置
REPO_ROOT = r"E:\ptrhon project\project2\vcmatch_repro\gitrepo1"
INPUT_CSV = r"E:\ptrhon project\project2\vcmatch_repro\dataset\test_cve_2012_0850.csv"
OUTPUT_CSV = r"E:\ptrhon project\project2\vcmatch_repro\dataset\detailed_cot_report.csv"

# CVE 描述
CVE_DESC = "The sbr_qmf_synthesis function in libavcodec/aacsbr.c in FFmpeg before 0.9.1 allows remote attackers to cause a denial of service (application crash) via a crafted mpg file that triggers memory corruption involving the v_off variable, probably a buffer underflow."


def get_git_data(repo_path, commit):
    """获取提交内容"""
    try:
        # 限制输出长度为 12000字符，留给思考空间
        cmd = ["git", "show", "--format=MSG:%s%n%b%nDIFF:", commit]
        res = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, errors='ignore', encoding='utf-8')
        output = res.stdout
        if len(output) > 12000:
            output = output[:12000] + "\n...[Truncated for Token Safety]"
        return output
    except:
        return ""


def analyze_with_cot(client, git_content):
    """
    【核心】思维链分析
    强制模型按步骤思考，并输出 JSON
    """
    prompt = f"""Target CVE: {CVE_DESC}

You are a security auditor. Analyze the Git Commit below.
Think step-by-step to determine if this is the security fix for the Target CVE.

Git Data:
{git_content}

# Output Requirements (JSON Format Only):
Please provide your response in the following JSON structure:
{{
    "step1_msg_analysis": "Analyze the commit message. Does it mention security keywords, CVE IDs, or specific bug IDs?",
    "step2_diff_analysis": "Analyze the code changes. Did it add boundary checks? Did it fix a buffer underflow/overflow? Did it remove unsafe functions?",
    "step3_cve_matching": "Compare the code changes with the CVE description. Do they match the vulnerability logic (e.g. 'v_off variable', 'buffer underflow')?",
    "final_decision": "YES" or "NO",
    "confidence": 0-100
}}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",  # 或 deepseek-chat
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        return {"final_decision": "NO", "step1_msg_analysis": f"Error: {e}"}


def main():
    # 1. 初始化
    if not BASE_URL:
        client = OpenAI(api_key=API_KEY)
    else:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    df = pd.read_csv(INPUT_CSV)
    results = []

    print(f"开始深度思维链分析: {len(df)} 条数据")

    # 2. 遍历
    for _, row in tqdm(df.iterrows(), total=len(df)):
        repo = row['repo']
        commit = row['commit']
        label = row['label']

        # 路径处理
        repo_path = os.path.join(REPO_ROOT, repo)
        if 'imagemagick' in repo.lower() and not os.path.exists(repo_path):
            repo_path = os.path.join(REPO_ROOT, 'ImageMagick6')

        if not os.path.exists(repo_path): continue

        # 获取数据 & 分析
        git_content = get_git_data(repo_path, commit)
        ai = analyze_with_cot(client, git_content)

        # 整理结果
        is_fix = 1 if str(ai.get("final_decision")).upper() == "YES" else 0

        results.append({
            "commit": commit,
            "ground_truth": label,
            "ai_pred": is_fix,
            "is_correct": (label == is_fix),
            # 详细思考过程
            "confidence": ai.get("confidence"),
            "thought_1_msg": ai.get("step1_msg_analysis"),
            "thought_2_diff": ai.get("step2_diff_analysis"),
            "thought_3_match": ai.get("step3_cve_matching")
        })

    # 3. 保存
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 统计
    correct = len(out_df[out_df['is_correct'] == True])
    print(f"\n[完成] 准确率: {correct}/{len(out_df)} ({correct / len(out_df) * 100:.1f}%)")
    print(f"详细思考报告已保存: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()