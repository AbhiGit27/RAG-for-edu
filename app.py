import streamlit as st
import google.generativeai as genai

genai.configure(api_key="API KEY WOULD BE HERE")
generation_config = genai.GenerationConfig(temperature=0.0)
model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)

def mock_get_context():
  return "Deadlock in an OS occurs when a process enters a waiting state because a requested system resource is held by another waiting process."

st.title(" 3rd Semester RAG Bot")
st.write("Welcome to your personal assistant for COA, OS, and PPS subjects.")

prompt= st.chat_input("Ask a question about your courses")
if prompt:
  with st.chat_message("user"):
        st.write(prompt)
  with st.chat_message("assistant"):
        with st.spinner("Searching course documents..."):

          retrieved_context = mock_get_context(prompt)
          strict_prompt = f"""
            You are a strict teaching assistant for a 3rd-semester computer science course.
            Answer the user's question using ONLY the provided context. 
            Do not use outside knowledge. If the answer isn't in the context, say exactly: "The course documents do not contain the answer to this question."
            
            Context from Course Documents:
            {retrieved_context}
            User Question: {prompt}
            """ 
          try:
            response = model.generate_content(strict_prompt)
            st.write(response.text)
          except Exception as e:
            st.error(f"Error connecting to gemini: {e}")
             
