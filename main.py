import os
import streamlit as st
import requests
from langchain_core.messages import AIMessage
from PyPDF2 import PdfReader
import pandas as pd
from io import BytesIO
from docx import Document
import matplotlib.pyplot as plt


TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
OCR_SPACE_API_KEY = st.secrets["OCR_SPACE_API_KEY"]



class llmClass:
    def __init__(self, api_key: str,
                 model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.together.xyz/v1/chat/completions"

    def invoke(self, messages: list[dict]) -> AIMessage:
        resp = requests.post(
            self.url,
            json={"model": self.model, "messages": messages},
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"}
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        return AIMessage(content=msg["content"])

if not (TOGETHER_API_KEY and OCR_SPACE_API_KEY):
    st.error("Please set TOGETHER_API_KEY and OCR_SPACE_API_KEY in your .env file......")
    st.stop()
llm = llmClass(api_key=TOGETHER_API_KEY)


def read_pdf(path):
    return "\n".join(
        page.extract_text() or "" for page in PdfReader(path).pages
    )

def read_txt(uploaded):
    return uploaded.getvalue().decode('utf-8', errors='ignore')

def read_csv(uploaded):
    return pd.read_csv(uploaded)

def read_xlsx(uploaded):
    return pd.read_excel(uploaded, engine='openpyxl')

def read_docx(path):
    return "\n".join(
        p.text for p in Document(path).paragraphs
    )

def read_image_ocr_space(uploaded):
    """
    Use OCR.Space API to extract text, specifying filename and filetype.
    """
    
    filename = uploaded.name if hasattr(uploaded, 'name') else 'image.png'
    mime = {
        'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'bmp': 'image/bmp'
    }
    ext = filename.split('.')[-1].lower()
    content_type = mime.get(ext, 'application/octet-stream')
    files = {
        'file': (filename, uploaded.getvalue(), content_type)
    }
    data = {
        'apikey': OCR_SPACE_API_KEY,
        'language': 'eng',
        'isOverlayRequired': False,
    }
    r = requests.post(
        "https://api.ocr.space/parse/image",
        files=files,
        data=data
    )
    r.raise_for_status()
    result = r.json()
    # Handle success
    if result.get('ParsedResults'):
        return result['ParsedResults'][0].get('ParsedText', '')
    # Handle errors gracefully
    err = result.get('ErrorMessage') or result.get('ErrorDetails')
    if isinstance(err, list):
        return '\n'.join(err)
    if err:
        return err
    return 'OCR failed: no text detected.'

#-------------------------------------------app------------------------------------------------------

st.set_page_config(page_title="Data Analyst Agent 🤖", layout="wide")
st.title("Multi‑File Data Analyst Agent")

uploaded = st.file_uploader(
    "Upload a file (PDF, DOCX, TXT, CSV, XLSX, Image):",
    type=["pdf","docx","txt","csv","xlsx","png","jpg","jpeg","bmp"]
)

if uploaded:
   
    if 'file_name' not in st.session_state or st.session_state.file_name != uploaded.name:
        st.session_state.clear()
        st.session_state.file_name = uploaded.name
        ext = uploaded.name.split('.')[-1].lower()
       
        if ext == 'pdf':
            with open('temp.pdf', 'wb') as f: f.write(uploaded.getbuffer())
            content = read_pdf('temp.pdf')
            df = None
        elif ext == 'docx':
            with open('temp.docx', 'wb') as f: f.write(uploaded.getbuffer())
            content = read_docx('temp.docx')
            df = None
        elif ext == 'txt':
            content = read_txt(uploaded)
            df = None
        elif ext == 'csv':
            df = read_csv(uploaded)
            content = df.to_csv(index=False)
        elif ext in ('xlsx','xls'):
            df = read_xlsx(uploaded)
            content = df.to_csv(index=False)
        elif ext in ('png','jpg','jpeg','bmp'):
            content = read_image_ocr_space(uploaded)
            df = None
        else:
            content, df = 'Unsupported file type.', None
      
        st.session_state.content = content
        st.session_state.df = df
        st.session_state.messages = [
            {'role':'system', 'content':f'You are a helpful data analysis assistant.\n\n{content}'}
        ]

    st.success(f"Loaded {st.session_state.file_name} successfully.")
    content = st.session_state.content
    df = st.session_state.df

    
    if df is not None:
        st.subheader('Visualization')
        st.dataframe(df, use_container_width=True)
        numeric = df.select_dtypes(include='number')
        if not numeric.empty:
            with st.form('viz_form'):
                ctype = st.selectbox('Chart:', ['line','bar','area','scatter','pie'])
                if ctype in ['line','bar','area']:
                    cols = st.multiselect('Columns:', numeric.columns.tolist(), default=numeric.columns.tolist()[:2])
                elif ctype == 'scatter':
                    x = st.selectbox('X-axis:', numeric.columns.tolist())
                    y = st.selectbox('Y-axis:', [c for c in numeric.columns if c!=x])
                else:
                    pie_col = st.selectbox('Pie Column:', df.columns.tolist())
                go = st.form_submit_button('Plot')
            if go:
                fig, ax = plt.subplots()
                if ctype == 'line':
                    for col in cols: ax.plot(df.index, df[col], label=col)
                    ax.legend()
                elif ctype == 'bar': df[cols].plot(kind='bar', ax=ax)
                elif ctype == 'area': df[cols].plot(kind='area', ax=ax)
                elif ctype == 'scatter': ax.scatter(df[x], df[y]); ax.set_xlabel(x); ax.set_ylabel(y)
                elif ctype == 'pie': df[pie_col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax); ax.set_ylabel('')
                st.pyplot(fig)
        else:
            st.info('No numeric data to visualize.')

  
    st.text_area('Content preview:', content[:1000], height=200)
    st.markdown('---')
    for m in st.session_state.messages[1:]:
        st.chat_message(m['role']).write(m['content'])
    q = st.chat_input('Ask follow-up question (please wait ~45–50s between requests):')
    if q:
        st.session_state.messages.append({'role':'user','content':q})
        with st.spinner('Thinking...'):
            ai = llm.invoke(st.session_state.messages)
        st.session_state.messages.append({'role':'assistant','content':ai.content})
        st.chat_message('assistant').write(ai.content)
else:
    st.info('Upload a supported file to begin.')
