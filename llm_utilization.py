import os
import requests
import time

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small")
API_URL = "https://api.mistral.ai/v1/chat/completions"

def call_mistral_chat(prompt, system_message="You are a helpful assistant.", max_tokens=256, temperature=0):
    """
    Simple function to call Mistral API and get response
    """
    
    # Check if API key is available
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY environment variable is not set")
    
    # Prepare the request
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        # Make the API call
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        
        # Check if request was successful
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        
        # Extract and return the response
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        return answer.strip()
        
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Please try again.")
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    
    except KeyError as e:
        raise Exception(f"Unexpected API response format: {str(e)}")
    
    except Exception as e:
        raise Exception(f"Error calling Mistral API: {str(e)}")

