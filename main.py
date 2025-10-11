# main.py - Greenlock Chatbot with Hybrid Search + Re-Ranking
import os
import uvicorn
import json
import hashlib
import pickle
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import time
import shutil
from typing import List, Dict, Tuple
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# BM25 for sparse retrieval
from rank_bm25 import BM25Okapi
import numpy as np

# Cross-Encoder for re-ranking
from sentence_transformers import CrossEncoder

# Web scraping
import requests
from bs4 import BeautifulSoup

load_dotenv()

app = FastAPI(title="Greenlock AI Chatbot with Hybrid Search + Re-Ranking")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "uploads"
WEB_CONTENT_DIR = "web_content"
VECTOR_DIR = "vectorstore"
CACHE_FILE = "web_cache.json"
BM25_FILE = "bm25_index.pkl"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(WEB_CONTENT_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# Greenlock website URLs to scrape
GREENLOCK_URLS = [
    "https://www.greenlock.io/", 
    "https://www.greenlock.io/insights/the-hidden-drains-on-family-office-portfolios",
    "https://www.greenlock.io/insights/racing-on-a-sinking-boat",
    "https://www.greenlock.io/insights/why-hedge-funds-dont-justify-their-fees",
    "https://www.greenlock.io/insights/evolution-of-private-equity-fees-from-1970-to-today",
    "https://www.greenlock.io/insights/private-equity-fees-catch-up",
    "https://www.greenlock.io/insights/private-equity-fees-carried-interest",
    "https://www.greenlock.io/insights/private-equity-fees-management-fee",
    "https://www.greenlock.io/insights/private-equity-fees-clawback",
    "https://www.greenlock.io/insights/banks-hide-up-to-50-percent-of-family-offices-portfolio-fees",
    "https://www.greenlock.io/insights/private-equity-fees-from-today-to-the-future",
    "https://www.greenlock.io/insights/family-office-fees-custody-advisory",
    "https://www.greenlock.io/insights/why-fees-matter",
    "https://www.greenlock.io/insights/investment-fees-benchmarks",
    "https://www.greenlock.io/insights/tip-of-an-iceberg",
    "https://www.greenlock.io/insights/multi-family-office-fee-and-cost",
]

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv("MODEL", "deepseek/deepseek-chat-v3"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    temperature=0.3,
    max_tokens=800
)

# LLM for query rewriting
query_rewriter_llm = ChatOpenAI(
    model=os.getenv("MODEL", "deepseek/deepseek-chat-v3"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    temperature=0.1,
    max_tokens=200
)

# Initialize Cross-Encoder Re-ranker
print("🧠 Loading Cross-Encoder re-ranker...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Re-ranker loaded successfully!")

# Storage
chat_history = []
knowledge_base_info = {
    "document_loaded": False,
    "website_scraped": False,
    "last_update": None,
    "total_chunks": 0,
    "sources": [],
    "hybrid_search_enabled": True,
    "query_rewriting_enabled": True,
    "reranking_enabled": True
}

# Global variables for hybrid search
bm25_index = None
bm25_documents = []

# Prompt templates
GREENLOCK_PROMPT = PromptTemplate(
    input_variables=["context", "question", "sources"],
    template="""You are Greenlock's AI assistant. Answer questions based on the provided context from Greenlock's documentation and website.

CONTEXT FROM GREENLOCK KNOWLEDGE BASE:
{context}

SOURCES USED:
{sources}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the Greenlock context provided above
2. Be concise, clear, and helpful
3. If information is from the document, mention "according to Greenlock documentation"
4. If information is from the website, mention "from Greenlock website"
5. If you're not sure or context doesn't contain the answer, say so honestly
6. Use a professional, friendly tone

ANSWER:"""
)

FALLBACK_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are Greenlock's AI assistant. The user asked: "{question}"

I couldn't find specific information about this in Greenlock's documentation or website.

Provide a brief, helpful response that:
1. Acknowledges you don't have specific information in the knowledge base
2. Suggests what the user might do (contact support, check website, etc.)
3. Keeps it brief and professional

ANSWER:"""
)

QUERY_REWRITE_PROMPT = PromptTemplate(
    input_variables=["original_query"],
    template="""You are a query optimization expert for Greenlock's investment fee analysis system.

ORIGINAL USER QUERY: "{original_query}"

Generate 3 optimized search queries that will help find relevant information about Greenlock's services, fees, investment insights, or documentation.

RULES:
1. Keep queries clear and searchable
2. Expand abbreviations and slang
3. Fix typos and grammar
4. Handle mixed language (Indonesian-English)
5. Make queries specific to Greenlock's domain (investment fees, portfolio management, etc.)
6. Each query should explore different aspects/keywords

FORMAT (return ONLY the 3 queries, one per line, no numbering):
<query 1>
<query 2>
<query 3>

QUERIES:"""
)

class CrossEncoderReRanker:
    """Cross-Encoder based re-ranker for document retrieval"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize re-ranker
        
        Args:
            model_name: HuggingFace cross-encoder model name
        """
        print(f"🔧 Initializing CrossEncoderReRanker with {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        print(f"✅ Re-ranker ready!")
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[Document] or List[Tuple[Document, float]]:
        """
        Re-rank documents using cross-encoder
        
        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of top documents to return
            return_scores: Whether to return scores with documents
            
        Returns:
            List of top-k re-ranked documents (with scores if return_scores=True)
        """
        if not documents:
            return []
        
        print(f"🔄 Re-ranking {len(documents)} documents for query: '{query[:50]}...'")
        
        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get scores from cross-encoder
        start_time = time.time()
        scores = self.model.predict(pairs)
        rerank_time = int((time.time() - start_time) * 1000)
        
        print(f"  ⏱️  Re-ranking took {rerank_time}ms")
        
        # Sort documents by score (descending)
        ranked_results = sorted(
            zip(documents, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Log top scores
        print(f"  📊 Top 5 scores:")
        for i, (doc, score) in enumerate(ranked_results[:5], 1):
            source = doc.metadata.get('source_name', 'Unknown')
            print(f"     {i}. Score: {score:.4f} - {source[:50]}")
        
        # Return top k
        top_results = ranked_results[:top_k]
        
        if return_scores:
            return top_results
        else:
            return [doc for doc, score in top_results]

# Initialize global re-ranker
reranker_instance = CrossEncoderReRanker()

def rewrite_query(original_query: str) -> List[str]:
    """Rewrite user query into multiple optimized variations using LLM"""
    try:
        print(f"🔄 Rewriting query: '{original_query}'")
        
        prompt = QUERY_REWRITE_PROMPT.format(original_query=original_query)
        result = query_rewriter_llm.invoke(prompt)
        response = result.content if hasattr(result, "content") else str(result)
        
        rewritten_queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        rewritten_queries = [q for q in rewritten_queries if 3 < len(q) < 200]
        rewritten_queries = rewritten_queries[:3]
        
        if original_query not in rewritten_queries:
            rewritten_queries.insert(0, original_query)
        
        print(f"  ✅ Generated {len(rewritten_queries)} query variations")
        for i, q in enumerate(rewritten_queries, 1):
            print(f"     {i}. {q}")
        
        return rewritten_queries
        
    except Exception as e:
        print(f"  ⚠️ Query rewriting failed: {e}")
        return [original_query]

def tokenize_text(text: str) -> List[str]:
    """Simple tokenization for BM25"""
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def build_bm25_index(documents: List[Document]):
    """Build BM25 index from documents"""
    global bm25_index, bm25_documents
    
    print("🔍 Building BM25 sparse index...")
    
    bm25_documents = documents
    tokenized_corpus = [tokenize_text(doc.page_content) for doc in documents]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    bm25_path = os.path.join(VECTOR_DIR, BM25_FILE)
    with open(bm25_path, 'wb') as f:
        pickle.dump({
            'bm25': bm25_index,
            'documents': bm25_documents
        }, f)
    
    print(f"  ✅ BM25 index built with {len(documents)} documents")

def load_bm25_index():
    """Load BM25 index from disk"""
    global bm25_index, bm25_documents
    
    bm25_path = os.path.join(VECTOR_DIR, BM25_FILE)
    if os.path.exists(bm25_path):
        try:
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
                bm25_index = data['bm25']
                bm25_documents = data['documents']
            print(f"✅ BM25 index loaded ({len(bm25_documents)} documents)")
            return True
        except Exception as e:
            print(f"❌ Error loading BM25 index: {e}")
            return False
    return False

def sparse_search(query: str, k: int = 10) -> List[Tuple[Document, float]]:
    """BM25 sparse retrieval"""
    if bm25_index is None or not bm25_documents:
        return []
    
    tokenized_query = tokenize_text(query)
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-k:][::-1]
    
    results = [
        (bm25_documents[idx], scores[idx])
        for idx in top_indices
        if scores[idx] > 0
    ]
    
    return results

def dense_search(vectorstore: FAISS, query: str, k: int = 10) -> List[Tuple[Document, float]]:
    """FAISS dense retrieval with scores"""
    if vectorstore is None:
        return []
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    results_with_scores = [
        (doc, 1 / (1 + distance))
        for doc, distance in results
    ]
    
    return results_with_scores

def hybrid_search(
    vectorstore: FAISS,
    query: str,
    k: int = 6,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5
) -> List[Document]:
    """Hybrid search combining dense (FAISS) and sparse (BM25) retrieval"""
    print(f"🔍 Hybrid Search: query='{query}', k={k}")
    
    dense_results = dense_search(vectorstore, query, k=k*2)
    sparse_results = sparse_search(query, k=k*2)
    
    print(f"  📊 Dense: {len(dense_results)} results")
    print(f"  📊 Sparse: {len(sparse_results)} results")
    
    doc_scores = {}
    
    if dense_results:
        max_dense_score = max(score for _, score in dense_results)
        for doc, score in dense_results:
            content_key = doc.page_content
            normalized_score = score / max_dense_score if max_dense_score > 0 else 0
            doc_scores[content_key] = {
                'doc': doc,
                'dense_score': normalized_score * dense_weight,
                'sparse_score': 0
            }
    
    if sparse_results:
        max_sparse_score = max(score for _, score in sparse_results) if sparse_results else 1
        for doc, score in sparse_results:
            content_key = doc.page_content
            normalized_score = score / max_sparse_score if max_sparse_score > 0 else 0
            
            if content_key in doc_scores:
                doc_scores[content_key]['sparse_score'] = normalized_score * sparse_weight
            else:
                doc_scores[content_key] = {
                    'doc': doc,
                    'dense_score': 0,
                    'sparse_score': normalized_score * sparse_weight
                }
    
    for key in doc_scores:
        doc_scores[key]['final_score'] = (
            doc_scores[key]['dense_score'] + 
            doc_scores[key]['sparse_score']
        )
    
    sorted_docs = sorted(
        doc_scores.values(),
        key=lambda x: x['final_score'],
        reverse=True
    )
    
    top_docs = [item['doc'] for item in sorted_docs[:k]]
    
    print(f"  ✅ Returned {len(top_docs)} hybrid results")
    
    return top_docs

def multi_query_search(
    vectorstore: FAISS,
    queries: List[str],
    k_per_query: int = 5,
    use_hybrid: bool = True,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5
) -> List[Document]:
    """Search with multiple query variations and aggregate results"""
    print(f"🔍 Multi-query search with {len(queries)} variations")
    
    all_docs = []
    seen_contents = set()
    
    for query in queries:
        try:
            if use_hybrid and bm25_index is not None:
                docs = hybrid_search(
                    vectorstore, 
                    query, 
                    k=k_per_query,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight
                )
            else:
                docs = vectorstore.similarity_search(query, k=k_per_query)
            
            for doc in docs:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_contents:
                    all_docs.append(doc)
                    seen_contents.add(content_hash)
        
        except Exception as e:
            print(f"  ⚠️ Search failed for query '{query}': {e}")
    
    print(f"  ✅ Found {len(all_docs)} unique documents from all queries")
    
    return all_docs[:15]  # Return top 15 for re-ranking

def fix_encoding(text: str) -> str:
    """Fix common encoding issues from web scraping"""
    if not text:
        return text
    
    replacements = {
        'â': '→',
        'â': '"',
        'â': '"',
        'â': "'",
        'â': '—',
        'â': '–',
        'Â': '',
        'â¢': '•',
        'â¦': '...',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def scrape_greenlock_website(urls: List[str]) -> List[Document]:
    """Scrape Greenlock website pages"""
    print(f"🌐 Starting to scrape {len(urls)} Greenlock pages...")
    documents = []
    cache = load_web_cache()
    
    for url in urls:
        try:
            print(f"  📄 Scraping: {url}")
            
            url_hash = hashlib.md5(url.encode()).hexdigest()
            if url_hash in cache:
                cached_data = cache[url_hash]
                if (time.time() - cached_data.get('timestamp', 0)) < 86400:
                    print(f"    ✅ Using cached version")
                    documents.append(Document(
                        page_content=cached_data['content'],
                        metadata=cached_data['metadata']
                    ))
                    continue
            
            response = requests.get(url, timeout=10)
            response.encoding = 'utf-8' 
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            
            title = soup.find('title')
            page_title = title.string if title else url.split('/')[-1]
            
            # FIX ENCODING HERE! Add this:
            page_title = fix_encoding(page_title)
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)
            
            clean_text = fix_encoding(clean_text)
            
            if len(clean_text) < 100:
                print(f"    ⚠️  Skipped (too short)")
                continue
            
            doc = Document(
                page_content=clean_text,
                metadata={
                    "source_type": "website",
                    "source_name": f"Greenlock Website - {page_title}",
                    "url": url,
                    "scraped_at": datetime.now().isoformat()
                }
            )
            documents.append(doc)
            
            cache[url_hash] = {
                'content': clean_text,
                'metadata': doc.metadata,
                'timestamp': time.time()
            }
            
            print(f"    ✅ Scraped successfully ({len(clean_text)} chars)")
            time.sleep(1)
            
        except Exception as e:
            print(f"    ❌ Error scraping {url}: {e}")
    
    save_web_cache(cache)
    print(f"✅ Scraped {len(documents)} pages successfully!")
    return documents

def load_web_cache() -> Dict:
    """Load web scraping cache"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_web_cache(cache: Dict):
    """Save web scraping cache"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def process_greenlock_document(filepath: str) -> List[Document]:
    """Process Greenlock PDF document"""
    print(f"📄 Processing Greenlock document: {filepath}")
    
    try:
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        
        filename = os.path.basename(filepath)
        
        for i, page in enumerate(pages):
            page.metadata.update({
                "source_type": "document",
                "source_name": f"Greenlock Documentation - {filename}",
                "filename": filename,
                "page": i + 1
            })
        
        print(f"  ✅ Loaded {len(pages)} pages from {filename}")
        return pages
        
    except Exception as e:
        print(f"  ❌ Error processing document: {e}")
        return []

def build_knowledge_base() -> FAISS:
    """Build complete knowledge base from documents + website with hybrid search"""
    print("\n" + "="*70)
    print("🔨 BUILDING GREENLOCK KNOWLEDGE BASE WITH HYBRID SEARCH + RE-RANKING")
    print("="*70)
    
    all_documents = []
    sources_info = []
    
    # Load PDFs
    print("📚 Loading all PDF documents from uploads folder...")
    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(UPLOAD_DIR, pdf_file)
        print(f"  📄 Processing: {pdf_file}")
        doc_pages = process_greenlock_document(pdf_path)
        if doc_pages:
            all_documents.extend(doc_pages)
            sources_info.append({
                "type": "document",
                "name": f"Greenlock Document - {pdf_file}",
                "pages": len(doc_pages)
            })
            knowledge_base_info["document_loaded"] = True
    
    # Scrape website
    web_docs = scrape_greenlock_website(GREENLOCK_URLS)
    if web_docs:
        all_documents.extend(web_docs)
        sources_info.append({
            "type": "website",
            "name": "Greenlock Website",
            "pages": len(web_docs),
            "urls": len(GREENLOCK_URLS)
        })
        knowledge_base_info["website_scraped"] = True
    
    if not all_documents:
        print("❌ No documents found to build knowledge base!")
        return None
    
    print(f"\n📚 Total documents collected: {len(all_documents)}")
    
    # Split documents
    print("✂️  Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
        length_function=len
    )
    
    chunks = splitter.split_documents(all_documents)
    print(f"  ✅ Created {len(chunks)} chunks")
    
    # Build BM25 index
    build_bm25_index(chunks)
    
    # Create FAISS vectorstore
    print("🧠 Creating FAISS vectorstore (dense)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save vectorstore
    vector_path = os.path.join(VECTOR_DIR, "faiss_index")
    vectorstore.save_local(vector_path)
    print(f"  ✅ Vectorstore saved to {vector_path}")
    
    # Update knowledge base info
    knowledge_base_info["total_chunks"] = len(chunks)
    knowledge_base_info["sources"] = sources_info
    knowledge_base_info["last_update"] = datetime.now().isoformat()
    knowledge_base_info["hybrid_search_enabled"] = True
    knowledge_base_info["reranking_enabled"] = True
    
    print("="*70)
    print("✅ ADVANCED KNOWLEDGE BASE BUILT SUCCESSFULLY!")
    print("   📊 Dense Search: FAISS vectorstore")
    print("   📊 Sparse Search: BM25 index")
    print("   📊 Re-ranking: Cross-Encoder (MiniLM-L-6-v2)")
    print("="*70 + "\n")
    
    return vectorstore

def load_vectorstore() -> FAISS:
    """Load existing vectorstore and BM25 index"""
    vector_path = os.path.join(VECTOR_DIR, "faiss_index")
    if os.path.exists(vector_path):
        try:
            vectorstore = FAISS.load_local(
                vector_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            load_bm25_index()
            
            return vectorstore
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            return None
    return None

def extract_sources_from_docs(docs: List[Document]) -> List[Dict]:
    """Extract source information from retrieved documents"""
    sources = []
    seen = set()
    
    for doc in docs:
        metadata = doc.metadata
        source_type = metadata.get("source_type", "unknown")
        
        if source_type == "document":
            key = f"doc_{metadata.get('page', 'unknown')}"
            if key not in seen:
                sources.append({
                    "type": "document",
                    "name": "Greenlock Documentation",
                    "page": metadata.get("page")
                })
                seen.add(key)
        
        elif source_type == "website":
            url = metadata.get("url", "")
            if url and url not in seen:
                # FIX ENCODING dari metadata yang tersimpan
                source_name = metadata.get("source_name", "Greenlock Website")
                source_name = fix_encoding(source_name)  # ← TAMBAH INI!
                
                sources.append({
                    "type": "website",
                    "name": source_name,  # Pakai yang udah di-fix
                    "url": url
                })
                seen.add(url)
    
    return sources

@app.get("/")
async def root():
    return {
        "system": "Greenlock AI Chatbot with Advanced Retrieval",
        "version": "3.0.0",
        "knowledge_base": {
            "document_loaded": knowledge_base_info.get("document_loaded", False),
            "website_scraped": knowledge_base_info.get("website_scraped", False),
            "total_chunks": knowledge_base_info.get("total_chunks", 0),
            "last_update": knowledge_base_info.get("last_update"),
            "sources": knowledge_base_info.get("sources", []),
            "hybrid_search_enabled": knowledge_base_info.get("hybrid_search_enabled", True),
            "reranking_enabled": knowledge_base_info.get("reranking_enabled", True)
        },
        "features": [
            "📄 2 Greenlock Documents",
            "🌐 15 Website Pages",
            "🔍 Hybrid Search (Dense + Sparse)",
            "🎯 Cross-Encoder Re-ranking",
            "🔄 Query Rewriting & Expansion",
            "🧠 Smart Source Attribution",
            "💬 Real-time Chat Support"
        ],
        "retrieval_pipeline": [
            "1️⃣ Query Rewriting (3 variations)",
            "2️⃣ Multi-query Hybrid Search (15 docs)",
            "3️⃣ Cross-Encoder Re-ranking (top 5)",
            "4️⃣ LLM Answer Generation"
        ],
        "models": {
            "embeddings": "all-MiniLM-L6-v2",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "llm": "deepseek-chat-v3"
        }
    }

@app.get("/health")
async def health_check():
    vectorstore = load_vectorstore()
    
    return {
        "status": "healthy",
        "components": {
            "llm": "✅ Ready",
            "embeddings": "✅ Ready",
            "vectorstore_faiss": "✅ Ready" if vectorstore else "⚠️ Not built",
            "bm25_index": "✅ Ready" if bm25_index else "⚠️ Not built",
            "reranker": "✅ Ready" if reranker_instance else "⚠️ Not loaded",
            "knowledge_base": "✅ Loaded" if knowledge_base_info.get("total_chunks", 0) > 0 else "⚠️ Empty"
        },
        "knowledge_base": knowledge_base_info
    }

@app.get("/status")
async def get_status():
    """Get chatbot status"""
    vectorstore = load_vectorstore()
    
    return {
        "ready": vectorstore is not None and bm25_index is not None,
        "knowledge_base": knowledge_base_info,
        "total_messages": len(chat_history),
        "message": "Ready with advanced retrieval!" if vectorstore else "Building knowledge base..."
    }

@app.post("/upload-greenlock-doc")
async def upload_greenlock_document(file: UploadFile = File(...)):
    """Upload Greenlock documentation PDF"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save file
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"📄 Greenlock document uploaded: {file.filename}")
        
        # Rebuild knowledge base
        vectorstore = build_knowledge_base()
        
        return {
            "message": "Greenlock document uploaded successfully!",
            "filename": file.filename,
            "knowledge_base": knowledge_base_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh-website")
async def refresh_website_content():
    """Re-scrape Greenlock website"""
    try:
        print("🔄 Refreshing Greenlock website content...")
        
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        
        vectorstore = build_knowledge_base()
        
        return {
            "message": "Website content refreshed successfully!",
            "knowledge_base": knowledge_base_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild-knowledge-base")
async def rebuild_knowledge_base():
    """Rebuild entire knowledge base"""
    try:
        vectorstore = build_knowledge_base()
        
        return {
            "message": "Knowledge base rebuilt successfully with re-ranking!",
            "knowledge_base": knowledge_base_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    use_hybrid: bool = Form(True),
    use_query_rewriting: bool = Form(True),
    use_reranking: bool = Form(True),
    rerank_top_k: int = Form(5),
    dense_weight: float = Form(0.5),
    sparse_weight: float = Form(0.5)
):
    """
    Ask a question with advanced retrieval pipeline
    
    Pipeline: Query Rewriting → Hybrid Search → Re-ranking → Answer
    """
    start_time = time.time()
    
    try:
        # Load vectorstore
        vectorstore = load_vectorstore()
        
        if not vectorstore:
            answer = "The knowledge base is not ready yet. Please wait while I load Greenlock's documentation and website content."
            
            chat_history.append({
                "type": "user",
                "content": question,
                "timestamp": time.time()
            })
            chat_history.append({
                "type": "assistant",
                "content": answer,
                "timestamp": time.time()
            })
            
            return {
                "question": question,
                "answer": answer,
                "sources": [],
                "status": "knowledge_base_not_ready"
            }
        
        # STEP 1: Query rewriting
        queries = [question]
        if use_query_rewriting:
            queries = rewrite_query(question)
            print(f"📝 Using {len(queries)} query variations")
        
        # STEP 2: Multi-query hybrid search (retrieve 15 docs)
        if len(queries) > 1:
            retrieved_docs = multi_query_search(
                vectorstore,
                queries,
                k_per_query=5,
                use_hybrid=use_hybrid,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )
        else:
            if use_hybrid and bm25_index is not None:
                retrieved_docs = hybrid_search(
                    vectorstore, 
                    question, 
                    k=15,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight
                )
            else:
                retrieved_docs = vectorstore.similarity_search(question, k=15)
        
        print(f"🔍 Retrieved {len(retrieved_docs)} documents before re-ranking")
        
        # STEP 3: Re-ranking (NEW!)
        if use_reranking and len(retrieved_docs) > rerank_top_k:
            print(f"🎯 Re-ranking {len(retrieved_docs)} docs → top {rerank_top_k}")
            retrieved_docs = reranker_instance.rerank(
                query=question,  # Use original question for re-ranking
                documents=retrieved_docs,
                top_k=rerank_top_k,
                return_scores=False
            )
            search_method = "hybrid_with_rewriting_and_reranking" if use_query_rewriting else "hybrid_with_reranking"
        else:
            # No re-ranking, just take top k
            retrieved_docs = retrieved_docs[:rerank_top_k]
            search_method = "hybrid_with_rewriting" if use_query_rewriting else "hybrid_only"
        
        print(f"📊 Final document count: {len(retrieved_docs)}")
        
        # Extract context and sources
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = extract_sources_from_docs(retrieved_docs)
        
        # Format sources for prompt
        sources_text = "\n".join([
            f"- {s['name']}" + (f" (Page {s['page']})" if 'page' in s else f" ({s.get('url', '')})")
            for s in sources
        ])
        
        # STEP 4: Generate answer
        if context and len(context.strip()) >= 100:
            prompt = GREENLOCK_PROMPT.format(
                context=context,
                question=question,
                sources=sources_text
            )
        else:
            prompt = FALLBACK_PROMPT.format(question=question)
            sources = []
        
        result = llm.invoke(prompt)
        answer = result.content if hasattr(result, "content") else str(result)
        
        # Save to history
        response_time = int((time.time() - start_time) * 1000)
        
        chat_history.append({
            "type": "user",
            "content": question,
            "timestamp": time.time()
        })
        chat_history.append({
            "type": "assistant",
            "content": answer,
            "sources": sources,
            "search_method": search_method,
            "rewritten_queries": queries if use_query_rewriting else None,
            "used_reranking": use_reranking,
            "timestamp": time.time(),
            "response_time_ms": response_time
        })
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "search_method": search_method,
            "rewritten_queries": queries if use_query_rewriting else None,
            "used_reranking": use_reranking,
            "num_docs_retrieved": len(retrieved_docs),
            "response_time_ms": response_time,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history")
async def get_chat_history(limit: int = 50):
    """Get chat history"""
    return {
        "total_messages": len(chat_history),
        "messages": chat_history[-limit:] if limit else chat_history
    }

class ConnectionManager:
    def __init__(self):
        self.active_connection: WebSocket = None
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connection = websocket
    
    def disconnect(self):
        self.active_connection = None
    
    async def send_message(self, message: str):
        if self.active_connection:
            await self.active_connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket chat endpoint with full advanced retrieval pipeline"""
    await manager.connect(websocket)
    
    try:
        await manager.send_message("Hello! I'm Greenlock's AI assistant with advanced search (Query Rewriting + Hybrid Search + Re-ranking). Ask me anything!")
        
        vectorstore = load_vectorstore()

        while True:
            question = await websocket.receive_text()
            start_time = time.time()
            
            chat_history.append({
                "type": "user",
                "content": question,
                "timestamp": time.time()
            })
            
            if not vectorstore:
                answer = "Knowledge base is loading. Please try again in a moment."
                search_method = "none"
            else:
                try:
                    # Full pipeline
                    queries = rewrite_query(question)
                    
                    retrieved_docs = multi_query_search(
                        vectorstore,
                        queries,
                        k_per_query=5,
                        use_hybrid=(bm25_index is not None)
                    )
                    
                    # Re-rank
                    if len(retrieved_docs) > 5:
                        retrieved_docs = reranker_instance.rerank(
                            query=question,
                            documents=retrieved_docs,
                            top_k=5
                        )
                    
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    sources = extract_sources_from_docs(retrieved_docs)
                    
                    sources_text = "\n".join([
                        f"- {s['name']}" + (f" (Page {s['page']})" if 'page' in s else "")
                        for s in sources
                    ])
                    
                    if context and len(context.strip()) >= 100:
                        prompt = GREENLOCK_PROMPT.format(
                            context=context,
                            question=question,
                            sources=sources_text
                        )
                    else:
                        prompt = FALLBACK_PROMPT.format(question=question)
                    
                    result = llm.invoke(prompt)
                    answer = result.content if hasattr(result, "content") else str(result)
                    search_method = "full_pipeline"
                    
                except Exception as e:
                    answer = f"Sorry, an error occurred: {str(e)}"
                    search_method = "error"
            
            response_time = int((time.time() - start_time) * 1000)
            chat_history.append({
                "type": "assistant",
                "content": answer,
                "search_method": search_method,
                "timestamp": time.time(),
                "response_time_ms": response_time
            })
            
            await manager.send_message(answer)

    except WebSocketDisconnect:
        manager.disconnect()

# ================================
# STARTUP
# ================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("\n" + "="*70)
    print("🚀 STARTING GREENLOCK AI CHATBOT - ADVANCED RETRIEVAL")
    print("="*70)
    
    vectorstore = load_vectorstore()
    
    if not vectorstore:
        print("\n⚠️  No existing knowledge base found.")
        print("📋 Please do one of the following:")
        print("   1. Upload Greenlock documents: POST /upload-greenlock-doc")
        print("   2. Build knowledge base: POST /rebuild-knowledge-base")
        print("   3. Place PDF files in uploads/ folder and restart")
    else:
        print("\n✅ Existing knowledge base loaded!")
        print(f"   📚 Total chunks: {knowledge_base_info.get('total_chunks', 0)}")
        print(f"   📄 Documents: {len([s for s in knowledge_base_info.get('sources', []) if s['type'] == 'document'])}")
        print(f"   🌐 Websites: {len([s for s in knowledge_base_info.get('sources', []) if s['type'] == 'website'])}")
        print(f"   🔍 Hybrid search: {knowledge_base_info.get('hybrid_search_enabled', False)}")
        print(f"   🎯 Re-ranking: {knowledge_base_info.get('reranking_enabled', False)}")
        print(f"   📊 BM25 index: {'✅ Loaded' if bm25_index else '⚠️ Not loaded'}")
        print(f"   📊 Cross-Encoder: {'✅ Ready' if reranker_instance else '⚠️ Not loaded'}")
    
    print("\n" + "="*70)
    print("✅ SYSTEM READY!")
    print("🌐 Access at: http://localhost:8000")
    print("📖 API Docs at: http://localhost:8000/docs")
    print("="*70 + "\n")

if __name__ == "__main__":
    print("="*70)
    print("🏢 GREENLOCK AI CHATBOT - ADVANCED RETRIEVAL SYSTEM")
    print("="*70)
    print("\n📚 Knowledge Base:")
    print("  • 2 Greenlock Documents (PDFs)")
    print("  • 15 Website Pages")
    print("\n🔍 Retrieval Pipeline:")
    print("  1️⃣  Query Rewriting (LLM-based, 3 variations)")
    print("  2️⃣  Multi-query Hybrid Search (Dense FAISS + Sparse BM25)")
    print("  3️⃣  Cross-Encoder Re-ranking (MiniLM-L-6-v2)")
    print("  4️⃣  Context Generation & LLM Answer")
    print("\n⚙️  Models:")
    print("  • Embeddings: all-MiniLM-L6-v2 (384 dim)")
    print("  • Re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  • LLM: deepseek-chat-v3")
    print("\n✨ Features:")
    print("  ✅ Smart query understanding")
    print("  ✅ Hybrid dense+sparse retrieval")
    print("  ✅ Precision re-ranking")
    print("  ✅ Configurable search parameters")
    print("  ✅ Real-time chat & WebSocket")
    print("  ✅ Source attribution")
    print("  🌍 Open CORS")
    print("\n📊 Expected Performance:")
    print("  • Retrieval: ~500-600ms")
    print("  • Re-ranking: ~100-150ms")
    print("  • LLM: ~200-300ms")
    print("  • Total: ~800-1000ms")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)