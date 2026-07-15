import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if GROQ_API_KEY:
    GROQ_API_KEY = GROQ_API_KEY.strip()
    if GROQ_API_KEY.startswith('GROQ_API_KEY='):
        GROQ_API_KEY = GROQ_API_KEY.split('=', 1)[1]
    # remove surrounding quotes if present
    if GROQ_API_KEY.startswith(('"', "'")) and GROQ_API_KEY.endswith(('"', "'")):
        GROQ_API_KEY = GROQ_API_KEY[1:-1]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_kwargs={"k": 2}
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    question = ''
    answer = None
    error = None

    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if not question:
            error = 'Please enter a question.'
        elif not GROQ_API_KEY:
            error = 'GROQ_API_KEY is missing. Add it to .env or export it to your environment.'
        else:
            try:
                answer = qa.run(question)
            except Exception as exc:
                import traceback

                # Safe masked key for logs
                masked = None
                if GROQ_API_KEY:
                    k = GROQ_API_KEY.strip()
                    masked = (k[:4] + '...' + k[-4:]) if len(k) > 8 else k

                # Print traceback and exception details to server logs (does not expose full key)
                print(f"[Groq Error] masked_key={masked} exception={exc!r}")
                traceback.print_exc()

                # If the exception has an HTTP response, try to show its body for debugging
                try:
                    resp = getattr(exc, 'response', None)
                    if resp is not None:
                        body = getattr(resp, 'text', None) or repr(resp)
                        print('[Groq Error] response body:', body)
                except Exception:
                    pass

                message = str(exc)
                if 'invalid_api_key' in message or 'Invalid API Key' in message or 'AuthenticationError' in message:
                    error = 'Groq authentication failed. Check that GROQ_API_KEY is valid and not expired. (See server logs for details.)'
                else:
                    error = f'Error: {message} (See server logs for details.)'

    return render_template('index.html', question=question, answer=answer, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=True)
