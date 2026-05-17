import streamlit as st
import requests

# Set up the page layout
st.set_page_config(page_title="Fraud Intelligence Dashboard", layout="wide")
st.title("Fraud Detection & Alert Dashboard")
st.markdown("---")

# Create two columns: Left for input, Right for alerts and results
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("1. New Transaction Entry")
    st.write("*(Simulating the OCR Extraction / Real-time feed)*")
    
    # Form inputs for the user
    tx_id = st.text_input("Transaction ID", value="TX-10042")
    sender = st.text_input("Sender Company Name", value="Mike")
    receiver = st.text_input("Receiver Company Name", value="Abibos")
    amount = st.number_input("Amount (USD)", value=10000.00)
    risk = st.slider("Base Risk Score (From legacy system)", 0.0, 100.0, 50.0)
    
    if st.button("Submit to AI Engine", type="primary"):
        # Package the data exactly how our FastAPI expects it
        payload = {
            "Id": tx_id,
            "sender_company_name": sender,
            "receiver_company_name": receiver,
            "amount": amount,
            "currency": "USD",
            "risk_score": risk
        }
        
        try:
            # Send the data to the local FastAPI server
            response = requests.post("http://127.0.0.1:8000/api/v1/process_transaction", json=payload)
            if response.status_code == 200:
                st.session_state['last_result'] = response.json()
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error("🚨 Could not connect to the API. Is your FastAPI server running?")

with col2:
    st.header("2. AI Intelligence Output")
    
    # Check if we have a result from the API
    if 'last_result' in st.session_state:
        res = st.session_state['last_result']
        
        # Display large metric numbers
        m1, m2 = st.columns(2)
        m1.metric(label="Transaction ID", value=res['transaction_id'])
        m2.metric(label="Final Calculated Risk", value=f"{res['final_risk_score']} / 100")
        
        # Display Visual Alerts based on the decision
        st.subheader("System Decision:")
        decision = res['decision']
        
        if decision == "AUTO_APPROVE":
            st.success("AUTO-APPROVED: No significant anomalies detected.")
        elif decision == "MANUAL_REVIEW":
            st.warning("MANUAL REVIEW REQUIRED: Suspicious pattern detected.")
        else:
            st.error("AUTO-REJECTED: High risk of fraud.")
            
        # Display the specific reasons (Sprint 9 Requirement)
        st.subheader("Alert Details & Reasons:")
        if res['reasons']:
            for reason in res['reasons']:
                st.write(f"- {reason}")
        else:
            st.write("- None")
            
    else:
        st.info("Waiting for incoming transactions... Submit a transaction on the left to see the AI analysis.")