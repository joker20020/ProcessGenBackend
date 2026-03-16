import fitz
import re
from typing import List


class TextProcessor(object):

    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)  # 打开PDF文件
        text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # 加载每一页
            text += page.get_text()  # 提取页面文字

        return text

    def split_markdown_for_rag(
        self,
        markdown_text: str,
        min_words: int = 50,
        include_subsections: bool = True
    ) -> List[str]:
        """
        将 Markdown 文档按标题结构分割为适合 RAG 检索的文本块。
        每个块包含完整的标题路径和内容。

        Args:
            markdown_text (str): 输入的 Markdown 文本。
            min_words (int): 每个块最少单词数（低于此阈值会尝试合并）。
            include_subsections (bool): 是否在父节中包含子节内容（True 为聚合式）。

        Returns:
            List[str]: 文本块列表，可直接用于 embedding。
        """
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        lines = [line.rstrip() for line in markdown_text.splitlines() if line.strip() or line == '\n']

        # 存储所有节（用于构建结构）
        sections = []  # 每项：{'level': int, 'heading': str, 'lines': [...]}

        current_section = None

        for line in lines:
            match = heading_pattern.match(line)
            if match:
                # 保存上一节
                if current_section is not None:
                    sections.append(current_section)
                # 开始新节
                level = len(match.group(1))
                heading = match.group(2).strip()
                current_section = {
                    'level': level,
                    'heading': heading,
                    'lines': [line]
                }
            else:
                if current_section is None:
                    # 文档开头无标题的内容，创建一个 level 0 的“根”节
                    current_section = {
                        'level': 0,
                        'heading': 'Introduction',
                        'lines': [line]
                    }
                else:
                    current_section['lines'].append(line)

        if current_section is not None:
            sections.append(current_section)

        # 构建带层级路径的文本块
        chunks = []

        for i, sec in enumerate(sections):
            if sec['level'] == 0 and not sec['lines']:
                continue

            # 获取标题路径：从当前节向上找父节
            path = []
            for j in range(i, -1, -1):
                if sections[j]['level'] < sec['level']:
                    path.append(sections[j]['heading'])
                elif sections[j]['level'] == sec['level'] and j < i:
                    break
            path.reverse()
            path.append(sec['heading'])
            title_path = " > ".join(path) if path else "Untitled"

            # 内容：当前节内容
            content_lines = sec['lines'][1:]  # 去掉标题行本身

            # 是否包含所有子节内容
            if include_subsections:
                child_lines = []
                for j in range(i + 1, len(sections)):
                    if sections[j]['level'] <= sec['level']:
                        break
                    child_lines.extend(sections[j]['lines'])
                content_lines.extend(child_lines)

            content = '\n'.join(content_lines).strip()

            # 构造 chunk 文本
            chunk_text = f"Section: {title_path}\n\n{content}".strip()

            # 简单的最小长度过滤（按单词数）
            if len(chunk_text.split()) >= min_words or i == 0:
                chunks.append(chunk_text)
            else:
                # 太短，尝试合并到前一个 chunk
                if chunks:
                    chunks[-1] += "\n\n" + chunk_text
                else:
                    chunks.append(chunk_text)

        return chunks

if __name__ == "__main__":

    p = TextProcessor()
    # 使用示例
    # pdf_path = r"E:\documents\pythonProjects\南京项目\material\工艺卡片.pdf"
    # extracted_text = p.extract_text_from_pdf(pdf_path)
    # print(extracted_text)  # 输出提取的文本
    # with open(r"pdf_output.txt", "a+", encoding="utf-8") as f:
    #     f.write(extracted_text)

    md_path = r"../data/工艺卡片.md"
    with open(md_path, "r+", encoding="utf-8") as f:
        content = f.read()
    chunks = p.split_markdown_for_rag(content, include_subsections=False)

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
        print("\n")