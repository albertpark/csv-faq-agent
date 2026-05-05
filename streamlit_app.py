import streamlit as st
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", 
    "tabulate",
    "langchain==0.3.25",
    "langchain-openai==0.3.16",
    "langchain-experimental==0.3.4",
    "langchain-community==0.3.24"
])

import pandas as pd
import os
import requests
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler

# command to run the streamlit page
# streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false

# Show title and description.
st.title("📄 Hello Agent - CSV FAQ Agent")
st.write(
    "Ask any question about our policies and FAQs — the AI will find the answer from the data. "
    "Enter your OpenAI API key below to get started."
)

# ==========================================
#  START: GET OPENAI API KEY
# ==========================================
# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # ==========================================
    #  PART 1: AUTOMATIC FILE DOWNLOADER
    # ==========================================
    DATASET_DIR = "datasets"

    if "files_downloaded" not in st.session_state:
        os.makedirs(DATASET_DIR, exist_ok=True)  # creates folder if it doesn't exist
        
        files_to_download = {
            "saas_docs.csv":         "https://raw.githubusercontent.com/albertpark/csv-faq-agent/refs/heads/main/datasets/saas_docs.csv",
            "credit_card_terms.csv": "https://raw.githubusercontent.com/albertpark/csv-faq-agent/refs/heads/main/datasets/credit_card_terms.csv",
            "hospital_policy.csv":   "https://raw.githubusercontent.com/albertpark/csv-faq-agent/refs/heads/main/datasets/hospital_policy.csv",
            "ecommerce_faqs.csv":    "https://raw.githubusercontent.com/albertpark/csv-faq-agent/refs/heads/main/datasets/ecommerce_faqs.csv"
        }
        
        print("--- Downloading Files from Github ---")
        for filename, url in files_to_download.items():
            filepath = os.path.join(DATASET_DIR, filename)
            if not os.path.exists(filename):
                r = requests.get(url)
                with open(filename, "wb") as f:
                    f.write(r.content)
                print(f"Downloaded: {filename}")
            else:
                print(f"Skipped: {filename} (Already exists)")
        print("--- Download Complete ---\n")

        st.session_state.files_downloaded = True
        st.session_state.files_to_download = files_to_download

    # ==========================================
    #  PART 2: LOAD CSVs + INIT AGENT (ONCE)
    # ==========================================
    if "agent" not in st.session_state:
        dataframes = [] # We will store all the loaded tables here
        loaded_names = []

        print("--- Loading Dataset Files ---")
        try:
            for filename in st.session_state.files_to_download.keys():
                filepath = os.path.join(DATASET_DIR, filename)
                df = pd.read_csv(filename)
                dataframes.append(df)
                loaded_names.append(filename)
                print(f"SUCCESS: Loaded '{filename}' ({len(df)} rows)")

        except Exception as e:
            print(f"\nERROR loading files: {e}")
            sys.exit()
        print("--- Loading Complete ---\n")
        
        # B. DEFINE THE RULES
        system_prompt = """
        You are a smart data assistant capable of reading multiple CSV files.
        - You have access to 4 different datasets: SaaS Docs, Credit Card Terms, Hospital Policy, and Ecommerce FAQs.
        - User can upload additional CSV files which you will append to the current datasets.
        - When asked a question, determine which DataFrame is most relevant.
        - Do NOT answer from general knowledge.
        - Answer in plain English.
        """

        try:
            # Initialize the LLM
            client = llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                api_key=openai_api_key
            )

            # Create the Pandas Agent
            agent = create_pandas_dataframe_agent(
                llm,
                dataframes,
                verbose=True,
                agent_type="openai-functions",
                allow_dangerous_code=True
            )

            st.session_state.agent = agent
            st.session_state.system_prompt = system_prompt
            print("AI Agent is ready!")
            print("Example: 'What is the visiting hour in the hospital?' or 'What is the API limit?'")

        except Exception as e:
            print(f"Error initializing agent: {e}")
            sys.exit()

    # ==========================================
    #  PART 3: CHAT RESPONSE
    # ==========================================
    
    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload the policies (.csv)", type=("csv")
    )

    # Ask the user for a question via `st.text_area`.
    user_input = st.text_area(
        "Now ask a question about the policy!",
        placeholder="What are the visiting hours in the hospital?"
        #disabled=not uploaded_file,
    )

    final_query = system_prompt + "\n\nQuestion: " + user_input
    print("AI is thinking...")

    try:
        # ---------------------------------------------------------
        # The result will be a dictionary, access ['output']
        # ---------------------------------------------------------
        if user_input:
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent.invoke(final_query, callbacks=[st_callback])['output']
            st.write(response)

        print(f"AI: {response}\n" + "-"*30)
    except Exception as e:
        print(f"An error occurred: {e}")

    # if uploaded_file and question:
    #     # Process the uploaded file and question.
    #     document = uploaded_file.read().decode()
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": f"Here's a document: {document} \n\n---\n\n {question}",
    #         }
    #     ]

    #     # Generate an answer using the OpenAI API.
    #     stream = client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=messages,
    #         stream=True,
    #     )

    #     # Stream the response to the app using `st.write_stream`.
    #     st.write_stream(stream)
