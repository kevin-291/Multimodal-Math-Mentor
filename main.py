import streamlit as st
import uuid
import os
import tempfile
from langgraph.types import Command
from graph import app as workflow_app 
from utils.ocr import OCR 
from utils.asr import ASR 

st.set_page_config(page_title="JEE Math AI Tutor", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "confidence" not in st.session_state:
    st.session_state.confidence = 1.0

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("🎓 JEE Math AI Tutor")
st.markdown("End-to-End Problem Solver with Human-in-the-Loop & Self-Learning")
run_workflow = False

input_mode = st.radio("Select Input Mode:", ["Text", "Image", "Audio"], horizontal=True)

with st.container(border=True):
    if input_mode == "Text":
        user_text = st.text_area("Type your math problem here:")
        if st.button("Submit & Solve", type="primary"):
            st.session_state.extracted_text = user_text
            st.session_state.edited_text = user_text
            st.session_state.confidence = 1.0 
            run_workflow = True 
            
    elif input_mode == "Image":
        uploaded_img = st.file_uploader("Upload an image of the problem", type=["png", "jpg", "jpeg"])
        if uploaded_img and st.button("Extract"):
            with st.spinner("Running OCR..."):
                try:
                    ocr_tool = OCR()
                    img_bytes = uploaded_img.read()
                    text, conf = ocr_tool.extract_text(img_bytes)
                    st.session_state.extracted_text = text
                    st.session_state.confidence = conf
                except Exception as e:
                    st.error(f"OCR Failed: {e}")
                    
    elif input_mode == "Audio":
        recorded_audio = st.audio_input("Record your math problem")
        
        if recorded_audio and st.button("Transcribe"):
            with st.spinner("Transcribing audio..."):
                try:
                    asr_tool = ASR()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(recorded_audio.read())
                        tmp_file_path = tmp_file.name
                    
                    text, conf = asr_tool.transcribe_audio(tmp_file_path)
                    
                    os.remove(tmp_file_path)
                    
                    st.session_state.extracted_text = text
                    st.session_state.confidence = conf
                    
                except Exception as e:
                    st.error(f"Audio Transcription Failed: {e}")

if st.session_state.extracted_text and input_mode in ["Image", "Audio"]:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_area(
            "Extraction Preview (Edit if needed):", 
            value=st.session_state.extracted_text, 
            key="edited_text"
        )
    with col2:
        conf_pct = int(st.session_state.confidence * 100)
        st.metric("Confidence Indicator", f"{conf_pct}%")
        if conf_pct < 70:
            st.warning("Low confidence. HITL review will be triggered.")
    
    if st.button("Run AI Workflow", type="primary"):
        run_workflow = True

if run_workflow and st.session_state.edited_text:
        initial_state = {
            "id": str(uuid.uuid4()),
            "text_input": st.session_state.edited_text,
            "confidence": st.session_state.confidence,
        }
        
        with st.status("Agent Trace: Running Workflow...", expanded=True) as status:
            for event in workflow_app.stream(initial_state, config=config):
                for node_name, state_update in event.items():
                    st.write(f"✅ **{node_name.upper()}** completed.")
            status.update(label="Workflow paused or completed!", state="complete", expanded=False)

current_state = workflow_app.get_state(config)

if current_state.next:
    st.error("🛑 Human Review Required!")
    
    interrupt_payload = current_state.tasks[0].interrupts[0].value
    
    with st.expander("View Reason for Interruption"):
        st.json(interrupt_payload) 
    
    with st.form("hitl_form"):
        st.subheader("Human-in-the-Loop Override")
        
        current_solution = interrupt_payload.get("current_state", {}).get("solution", "")
        edited_solution = st.text_area("Edit Solution (if applicable):", value=current_solution)
        
        feedback = st.text_input("Reviewer Feedback/Message:")
        
        col_a, col_b, col_c = st.columns(3)
        approve = col_a.form_submit_button("✅ Approve As-Is")
        edit = col_b.form_submit_button("✏️ Submit Edited Solution")
        reject = col_c.form_submit_button("❌ Reject Completely")
        
        decision = None
        if approve: decision = "approve"
        if edit: decision = "edit"
        if reject: decision = "reject"
        
        if decision:
            workflow_app.invoke(
                Command(resume={"type": decision, "edited_solution": edited_solution, "message": feedback}),
                config=config
            )
            st.rerun()

if not current_state.next and current_state.values.get("solution"):
    state_vals = current_state.values
    
    st.divider()
    st.header("🎯 Final Results")
    
    with st.expander("View Retrieved Context & Semantic Memory"):
        st.markdown("**RAG Context:**")
        st.write(state_vals.get("retrieved_context", "No context retrieved."))
        st.markdown("**Past Similar Solutions (Memory):**")
        st.write(state_vals.get("past_similar_problems", "No past similarities found."))
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.subheader("Raw Mathematical Solution")
        st.markdown(rf"**Solution:** {state_vals.get('solution', 'No solution generated.')}")
        
    with col_res2:
        st.subheader("Tutor Explanation")
        st.markdown(rf"**Explanation:** {state_vals.get('tutor_explanation', 'No explanation generated.')}") 
        
    st.divider()
    st.markdown("### Rate this answer")
    f_col1, f_col2 = st.columns([1, 4])
    
    with f_col1:
        if st.button("✅ Correct"):
            st.success("Thanks for the feedback! Positive learning signal stored.")
            
    with f_col2:
        with st.popover("❌ Incorrect"):
            user_comment = st.text_input("What went wrong?")
            if st.button("Submit Comment"):
                st.warning(f"Feedback logged: {user_comment}")
                
    if st.button("Start New Problem"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.extracted_text = ""
        st.session_state.confidence = 1.0
        st.rerun()