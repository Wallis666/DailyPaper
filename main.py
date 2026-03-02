import datetime
import json
import os
from typing import List, Dict

import arxiv
import google.genai as genai
import requests


# 配置部分：在此处填入 Google Gemini API Key 和 Push Plus Token
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PUSH_PLUS_TOKEN = os.getenv("PUSH_PLUS_TOKEN")

client = genai.Client(api_key=GOOGLE_API_KEY)


def get_latest_papers(
    topic: str = "Large Language Models",
    max_results: int = 3
) -> List[Dict[str, str]]:
    """
    从 ArXiv 获取指定主题的最新论文
    """
    print(f"正在检索关于 {topic} 的最新论文...")

    # 构建查询：按提交时间倒序排列
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers_data = []
    arxiv_client = arxiv.Client()
    for result in arxiv_client.results(search):
        papers_data.append({
            "title": result.title,
            "abstract": result.summary,
            "url": result.entry_id,
            "published": result.published
        })

    return papers_data


def generate_summary(paper: Dict[str, str]) -> str:
    """
    调用 Gemini API 生成中文解读
    """
    print(f"正在研读论文：{paper['title']} ...")

    # 植入设计好的 Prompt
    prompt = f"""
    # Role Assignment
    你是一位拥有 20 年经验的资深人工智能领域研究员，擅长快速阅读英文学术文献，并将其核心价值转化为逻辑严密、通俗易懂的中文技术简报。你对 Transformer 架构、多模态大模型以及强化学习等前沿技术有深刻的理解。

    # Task Description
    请阅读我提供的学术论文内容（或摘要），输出一份结构化的中文研报。你的目标是帮助读者在 1 分钟内准确判断该论文的价值，并掌握其核心创新点。

    # Constraints
    1. 必须使用中文进行输出，保留必要的英文专业术语（如 Zero-shot, Chain of Thought 等）。
    2. 严禁直接翻译原文摘要，必须基于理解进行重述和概括。
    3. 语气保持客观、专业，避免使用营销式夸张词汇。
    4. "创新点"部分必须具体，指出该论文解决了什么具体痛点，不仅是罗列功能。

    # Output Format
    请严格按照以下 Markdown 格式输出：

    #### 📄 论文标题：[论文的中文翻译标题]
    **原标题**：[Original English Title]

    **第一作者**：[First Author Name] | **机构**：[Institution]

    ##### 🎯 核心摘要
    [在此处撰写 150-200 字的中文摘要。主要描述论文的背景问题、提出的方法论以及最终达成的效果。]

    ##### 💡 核心创新点与贡献
    * **[创新点 1 名称]**：详细解释该创新的技术原理或实现方式，以及它相对于现有 SOTA 方法的优势。
    * **[创新点 2 名称]**：描述该方法在实验设计或数据集构建上的独特之处。
    * **[创新点 3 名称]**：总结该论文在实验结果上的突破（需包含具体的提升数据，如 Accuracy 提升了 x%）。

    ##### 🧐 简评与启示
    [用一句话总结该论文对当前研究领域的潜在影响或实际应用价值。]

    ---
    
    # Input Data
    Title: {paper["title"]}
    Abstract: {paper["abstract"]}
    URL: {paper["url"]}
    Published: {paper["published"]}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"解读失败，错误信息：{e}"


def main():
    # 1. 获取论文
    # 可以修改 topic 参数来关注不同的领域
    papers = get_latest_papers(topic="Multi-Agent System", max_results=2)

    title = f"📅 AI 前沿论文日报 ({datetime.date.today()})"
    content = ""

    # 2. 逐篇处理
    for paper in papers:
        summary = generate_summary(paper)

        # 拼接内容
        content += f"{summary}\n"
        content += f"##### 🔗 **原文链接**: {paper['url']}\n"
        content += "---\n\n"

    # 3. 输出结果
    url = "http://www.pushplus.plus/send"
    data = {
        "token": PUSH_PLUS_TOKEN,
        "title": title,
        "content": content,
        "template": "markdown"
    }
    body = json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    requests.post(url, data=body, headers=headers)


if __name__ == "__main__":
    main()