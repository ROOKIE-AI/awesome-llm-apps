"""
RAG即服务

使用Ragie和 OpenAI 构建的RAG即服务应用程序。


"""


import streamlit as st
import openai
import requests
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse

class RAGPipeline:
    def __init__(self, openai_api_key: str, ragie_api_key: str, base_url: str):
        """
        使用API密钥初始化RAG管道。
        """
        openai.api_key = openai_api_key
        self.ragie_api_key = ragie_api_key
        self.base_url = base_url
        
        # API端点
        self.RAGIE_UPLOAD_URL = f"{self.base_url}/documents/url"
        self.RAGIE_RETRIEVAL_URL = f"{self.base_url}/retrievals"
    
    def upload_document(self, url: str, name: Optional[str] = None, mode: str = "fast") -> Dict:
        """
        从URL上传文档到Ragie。
        """
        if not name:
            name = urlparse(url).path.split('/')[-1] or "document"
            
        payload = {
            "mode": mode,
            "name": name,
            "url": url
        }
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.ragie_api_key}"
        }
        
        response = requests.post(self.RAGIE_UPLOAD_URL, json=payload, headers=headers)
        
        if not response.ok:
            raise Exception(f"文档上传失败: {response.status_code} {response.reason}")
            
        return response.json()
    
    def retrieve_chunks(self, query: str, scope: str = "tutorial") -> List[str]:
        """
        从Ragie检索与查询相关的块。
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.ragie_api_key}"
        }
        
        payload = {
            "query": query,
            "filters": {
                "scope": scope
            }
        }
        
        response = requests.post(
            self.RAGIE_RETRIEVAL_URL,
            headers=headers,
            json=payload
        )
        
        if not response.ok:
            raise Exception(f"检索失败: {response.status_code} {response.reason}")
            
        data = response.json()
        return [chunk["text"] for chunk in data["scored_chunks"]]

    def create_system_prompt(self, chunk_texts: List[str]) -> str:
        """
        使用检索到的块创建系统提示。
        """
        return f"""这些非常重要: 你是"Ragie AI"，一个专业但友好的AI聊天机器人，作为用户的助手。你当前的任务是根据以下所有可用信息帮助用户。回答时要非正式、直接和简洁，不要有标题或问候语，但要包括所有相关内容。在适当的时候使用富文本Markdown，包括粗体、斜体、段落和列表。如果使用LaTeX，请使用双$$作为分隔符而不是单$。使用$$...$$而不是括号。在适当的时候将信息组织成多个部分或要点。不要包括原始项目ID或其他原始字段，除非用户要求。不要使用XML或其他标记，除非用户要求。以下是所有可用的信息来回答用户: === {chunk_texts} === 如果用户要求搜索但没有结果，请确保让用户知道你找不到任何东西，并告诉他们可能需要做些什么来找到所需的信息。结束系统指令"""

    def generate_response(self, system_prompt: str, query: str) -> str:
        """
        使用OpenAI生成响应。
        """
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=system_prompt + "\n\n用户: " + query,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7
        )
        
        return response.choices[0].text.strip()

    def process_query(self, query: str, scope: str = "tutorial") -> str:
        """
        通过完整的RAG管道处理查询。
        """
        chunks = self.retrieve_chunks(query, scope)
        
        if not chunks:
            return "未找到与您的查询相关的信息。"
        
        system_prompt = self.create_system_prompt(chunks)
        return self.generate_response(system_prompt, query)

def initialize_session_state():
    """初始化会话状态变量。"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'document_uploaded' not in st.session_state:
        st.session_state.document_uploaded = False
    if 'api_keys_submitted' not in st.session_state:
        st.session_state.api_keys_submitted = False

def main():
    st.set_page_config(page_title="RAG即服务", layout="wide")
    initialize_session_state()
    
    st.title(":linked_paperclips: RAG即服务")
    
    # API密钥部分
    with st.expander("🔑 API密钥配置", expanded=not st.session_state.api_keys_submitted):
        col1, col2, col3 = st.columns(3)
        with col1:
            ragie_key = st.text_input("Ragie API密钥", type="password", key="ragie_key")
        with col2:
            openai_key = st.text_input("OpenAI API密钥", type="password", key="openai_key")
        with col3:
            base_url = st.text_input("Ragie Base URL", value="https://api.ragie.ai", key="base_url")
        
        if st.button("提交API密钥"):
            if ragie_key and openai_key and base_url:
                try:
                    st.session_state.pipeline = RAGPipeline(openai_key, ragie_key, base_url)
                    st.session_state.api_keys_submitted = True
                    st.success("API密钥配置成功！")
                except Exception as e:
                    st.error(f"配置API密钥时出错: {str(e)}")
            else:
                st.error("请提供所有必需的API密钥和Base URL。")
    
    # 文档上传部分
    if st.session_state.api_keys_submitted:
        st.markdown("### 📄 文档上传")
        doc_url = st.text_input("输入文档URL")
        doc_name = st.text_input("文档名称（可选）")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            upload_mode = st.selectbox("上传模式", ["快速", "准确"])
        
        if st.button("上传文档"):
            if doc_url:
                try:
                    with st.spinner("正在上传文档..."):
                        st.session_state.pipeline.upload_document(
                            url=doc_url,
                            name=doc_name if doc_name else None,
                            mode=upload_mode
                        )
                        time.sleep(5)  # 等待索引
                        st.session_state.document_uploaded = True
                        st.success("文档上传并成功索引！")
                except Exception as e:
                    st.error(f"上传文档时出错: {str(e)}")
            else:
                st.error("请提供文档URL。")
    
    # 查询部分
    if st.session_state.document_uploaded:
        st.markdown("### 🔍 查询文档")
        query = st.text_input("输入您的查询")
        
        if st.button("生成响应"):
            if query:
                try:
                    with st.spinner("正在生成响应..."):
                        response = st.session_state.pipeline.process_query(query)
                        st.markdown("### 响应:")
                        st.markdown(response)
                except Exception as e:
                    st.error(f"生成响应时出错: {str(e)}")
            else:
                st.error("请输入查询内容。")

if __name__ == "__main__":
    main()