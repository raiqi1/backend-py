# test_openrouter.py
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def test_openrouter_connection():
    try:
        llm = ChatOpenAI(
            model=os.getenv("MODEL"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            temperature=0.7
        )
        
        print("Testing OpenRouter connection...")
        print(f"Model: {os.getenv('MODEL')}")
        print(f"Base URL: {os.getenv('OPENROUTER_BASE_URL')}")
        print(f"API Key: {os.getenv('OPENROUTER_API_KEY')[:20]}...")
        
        result = llm.invoke("Halo, apa kabar? Jawab dalam bahasa Indonesia.")
        
        if hasattr(result, 'content'):
            answer = result.content
        else:
            answer = str(result)
            
        print(f"\n✅ SUCCESS! Response: {answer}")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    test_openrouter_connection()