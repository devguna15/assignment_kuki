from retriever import get_relevant_faqs
from llm_utilization import call_mistral_chat

def create_prompt(user_question, faqs):
    """Create a simple prompt for the LLM"""
    
    # If no relevant FAQs found
    if not faqs:
        return f"""You are an e-commerce support agent. The customer asked: "{user_question}"

This question doesn't match our FAQ database. Politely explain that you specialize in e-commerce support and ask them to rephrase their question about orders, returns, products, or payments.

Keep your response to 3-4 sentences."""
    
    # Build FAQ context
    context = ""
    for i, faq in enumerate(faqs, 1):
        context += f"{i}. Q: {faq['Customer_Question']}\n   A: {faq['Detailed_Response']}\n\n"
    
    # Create the main prompt
    prompt = f"""You are a helpful e-commerce support agent. Answer the customer's question using the FAQ information below.

IMPORTANT: Keep your answer to 3-4 sentences maximum. Be direct and helpful.

FAQ Information:
{context}

Customer Question: {user_question}

Your Answer:"""
    
    return prompt

def generate_answer(user_question):
    """Generate an answer for the user's question"""
    
    # Validate input
    if not user_question or not user_question.strip():
        return "Please ask a valid question.", []
    
    if len(user_question.strip()) < 3:
        return "Please ask a more detailed question.", []
    
    try:
        # Get relevant FAQs
        faqs = get_relevant_faqs(user_question, top_k=3)
        
        # Create prompt
        prompt = create_prompt(user_question, faqs)
        
        # Get answer from LLM
        system_message = "You are a professional e-commerce support agent. Always keep responses brief and helpful."
        answer = call_mistral_chat(prompt, system_message=system_message, max_tokens=150, temperature=0)
        
        # Clean up the answer
        answer = answer.strip()
        if not answer:
            return "I'm having trouble generating an answer. Please try rephrasing your question.", faqs
        
        return answer, faqs
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}", []


