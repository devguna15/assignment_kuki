import streamlit as st
from retriever import get_relevant_faqs
from llm_utilization import call_mistral_chat

# Page setup
st.set_page_config(page_title="E-commerce Support", page_icon="🛒")
st.title("🛒 Customer Support Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if user_question := st.chat_input("Ask your question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)
    
    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get relevant FAQs
                faqs = get_relevant_faqs(user_question, top_k=3)
                
                # Build context
                context = ""
                for faq in faqs:
                    context += f"Q: {faq['Customer_Question']}\nA: {faq['Detailed_Response']}\n\n"
                
                # Create prompt
                prompt = f"""You are a helpful e-commerce support agent. Answer the customer's question using the FAQ context below. 
Keep your answer to 2-3 sentences maximum. Be direct and helpful.

FAQ Context:
{context}

Customer Question: {user_question}

Answer:"""
                
                # Get response
                answer = call_mistral_chat(prompt)
                st.write(answer)
                
                # Save assistant message
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Show FAQ context used
                if faqs:
                    with st.expander("FAQ Context Used"):
                        for i, faq in enumerate(faqs, 1):
                            st.write(f"**{i}. {faq['Customer_Question']}**")
                            st.write(f"*Answer: {faq['Detailed_Response']}*")
                            st.write(f"*Category: {faq['Product_Category']}*")
                            st.write("---")
                
            except Exception as e:
                error_msg = "Sorry, I'm having trouble right now. Please try again."
                st.write(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
