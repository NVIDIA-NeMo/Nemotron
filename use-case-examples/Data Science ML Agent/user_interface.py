import streamlit as st
import pandas as pd
import tempfile
import os
import io
from contextlib import redirect_stdout, redirect_stderr
import time
from functools import wraps

from src.chat_agent import ChatAgent

# ============= ADD TIMING DECORATOR HERE =============
def track_execution_time(method_name):
    """Decorator to track and display execution time for ML methods"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Print start message (will be captured)
            print(f"Starting {method_name}...")
            
            # Execute the original method
            result = func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Format time display
            if execution_time < 60:
                time_str = f"{execution_time:.2f} seconds"
            else:
                minutes = int(execution_time // 60)
                seconds = execution_time % 60
                time_str = f"{minutes} min {seconds:.1f} sec"
            
            # Print execution time (will be captured by redirect_stdout)
            print(f"\nTotal execution time for {method_name}: {time_str}")
            print(f"{method_name} completed successfully!")
            
            return result
        return wrapper
    return decorator

# Monkey-patch the ChatAgent methods to add timing
if "timing_patched" not in st.session_state:
    # Patch the underscore versions (these are what usually get called)
    if hasattr(ChatAgent, '_train_classification'):
        original = ChatAgent._train_classification
        ChatAgent._train_classification = track_execution_time('Classification Training')(original)
    
    if hasattr(ChatAgent, '_train_regression'):
        original = ChatAgent._train_regression
        ChatAgent._train_regression = track_execution_time('Regression Training')(original)
    
    if hasattr(ChatAgent, '_optimize'):
        original = ChatAgent._optimize
        ChatAgent._optimize = track_execution_time('Model Optimization')(original)
    
    st.session_state.timing_patched = True
# ============= END OF TIMING DECORATOR =============


st.set_page_config(page_title="ML Agent", page_icon="🤖", layout="wide")

# Initialize session state
if "ml_agent" not in st.session_state:
    st.session_state.ml_agent = ChatAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# Layout
left_col, right_col = st.columns([3, 7])

with left_col:
    st.header("🤖 ML Agent")
    
    # Multi-file upload
    uploaded_files = st.file_uploader(
        "Upload Datasets", 
        type=['csv', 'parquet'], 
        accept_multiple_files=True,
        help="Upload training data, test data, or any CSV/Parquet files"
    )
    
    # Process uploaded files
    if uploaded_files:
        st.subheader("📁 Uploaded Files")
        
        for uploaded_file in uploaded_files:
            file_key = uploaded_file.name
            
            # Save file if not already saved
            if file_key not in st.session_state.uploaded_files:
                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Store file info
                st.session_state.uploaded_files[file_key] = {
                    'path': tmp_path,
                    'name': uploaded_file.name
                }
                
                # Load and show basic info
                try:
                    if uploaded_file.name.endswith('.parquet'):
                        df = pd.read_parquet(tmp_path)
                    else:
                        df = pd.read_csv(tmp_path)
                    
                    st.session_state.uploaded_files[file_key]['shape'] = df.shape
                    st.session_state.uploaded_files[file_key]['columns'] = list(df.columns)
                    st.session_state.ml_agent.uploaded_files = getattr(st.session_state.ml_agent, 'uploaded_files', {})
                    st.session_state.ml_agent.uploaded_files[uploaded_file.name] = tmp_path
                
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                    continue
        
        # Display file info
        for file_key, file_info in st.session_state.uploaded_files.items():
            if 'shape' in file_info:
                shape = file_info['shape']
                st.write(f"**{file_info['name']}**")
                st.write(f"   Shape: {shape[0]:,} rows × {shape[1]} columns")
                
                # Show columns in expander
                with st.expander(f"Columns in {file_info['name']}", expanded=False):
                    st.write(", ".join(file_info['columns']))
    
    # Model status
    model_exists = os.path.exists("best_model.joblib")
    if model_exists:
        st.success("✅ Trained model available")
    else:
        st.info("ℹ️ No trained model yet")
    
    # Quick examples
    with st.expander("💡 Example Commands", expanded=False):
        st.markdown("""
        **Data Loading:**
        ```
        load_dataset(path, target)
        - Load CSV/Parquet file (target optional)
        set_target(column)
        - Set target column
        describe_data()
        - Show dataset information
        preview_data(rows?)
        - Show first few rows
        ```

        **Model Training:**
        ```
        train_classification()
        - Train classification models
        train_regression()
        - Train regression models
        ```

        **Classification Optimization:**
        ```
        optimize_logistic(trials?)
        - Optimize logistic regression
        optimize_svc(trials?)
        - Optimize support vector classifier
        optimize_forest(trials?)
        - Optimize random forest classifier
        ```

        **Regression Optimization:**
        ```
        optimize_ridge(trials?)
        - Optimize ridge regression
        optimize_forest_regressor(trials?)
        - Optimize random forest regressor
        optimize_svr(trials?)
        - Optimize support vector regressor
        ```

        **Predictions:**
        ```
        predict(test_data_path, output_path?)
        - Make predictions on test data
        ```

        **Results:**
        ```
        show_best_model(metric?)
        - Show best model
        show_history(limit?)
        - Show recent experiments
        ```

        **Examples:**
        ```
        "load dataset Titanic-Dataset.csv/Titanic-Dataset-test.csv"
        "set target variable to be 'Survived'"
        "train classification/regression model"
        "optimize svc with 50 trials"
        "optimize forest regressor with 30 trials"
        "show best model by r2"
        "make inference for the test dataset"
        ```
        """)
                    
with right_col:
    st.header("💬 Chat")
    
    # Control buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("📊 Show History"):
            response = st.session_state.ml_agent._show_history()
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user" and "paraphrased" in message:
                st.markdown(f"**Original:** {message['content']}")
                st.markdown(f"**Understood as:** _{message['paraphrased']}_")
            else:
                st.markdown(message["content"])
    
    # Initialize flags
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None

    # Show input only if not processing
    if not st.session_state.is_processing:
        user_input = st.chat_input("Ask about your data or give ML commands...")
        
        if user_input:
            # Store input and set processing flag
            st.session_state.pending_input = user_input
            st.session_state.is_processing = True
            st.rerun()  # Immediately rerun to hide the input
    else:
        # Don't show any input widget during processing
        pass
        
        # Process the pending input
        if st.session_state.pending_input:
            user_input = st.session_state.pending_input
            st.session_state.pending_input = None

            # ADD START TIME HERE
            start_time = time.time()
            
            # Add user message without paraphrase initially
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Create placeholders for dynamic updates
            user_msg_container = st.empty()
            assistant_msg_container = st.empty()

            # Show initial user message
            with user_msg_container.container():
                with st.chat_message("user"):
                    st.markdown(f"**Original:** {user_input}")
                    paraphrase_placeholder = st.empty()
                    paraphrase_placeholder.markdown("**Understood as:** _Analyzing..._")

            # Show initial assistant message
            with assistant_msg_container.container():
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    response_placeholder.markdown("⏳ Processing your question...")

            captured_output = io.StringIO()
            try:
                # Get paraphrased question and response
                with redirect_stdout(captured_output), redirect_stderr(captured_output):
                    paraphrased_question, response = st.session_state.ml_agent.chat(user_input)

                # Stream the paraphrased question
                paraphrase_words = paraphrased_question.split()
                for i in range(1, len(paraphrase_words) + 1):
                    partial_paraphrase = " ".join(paraphrase_words[:i])
                    paraphrase_placeholder.markdown(f"**Understood as:** _{partial_paraphrase}_")
                    time.sleep(0.03)  # Adjust speed for paraphrase streaming
                
                time.sleep(0.3)  # Brief pause to show the paraphrase

                # Update assistant message with typing effect
                response_placeholder.markdown("💭 Generating response...")
                time.sleep(0.3)
                
                # Combine with terminal output if any
                terminal_output = captured_output.getvalue().strip()
                if terminal_output:
                    full_response = response + "\n\n**Output:**\n```\n" + terminal_output + "\n```"
                else:
                    full_response = response

               # ADD END TIME AND CALCULATE DURATION HERE
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Format time display
                if execution_time < 60:
                    time_str = f"{execution_time:.2f} seconds"
                else:
                    minutes = int(execution_time // 60)
                    seconds = execution_time % 60
                    time_str = f"{minutes} min {seconds:.1f} sec"
                
                # Add timing to the response
                full_response += f"\n\n---\n⏱️ **Total processing time: {time_str}**"
                
                # For streaming effect
                words = full_response.split()
                for i in range(1, len(words) + 1):
                    partial_response = " ".join(words[:i])
                    response_placeholder.markdown(partial_response)
                    time.sleep(0.02)  # Adjust speed as needed
                
                # Save paraphrase + response to messages
                st.session_state.messages[-1]["paraphrased"] = paraphrased_question
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                try:
                    response = str(e)
                    if "too many values to unpack" not in response and "not enough values to unpack" not in response:
                        raise e
                    with redirect_stdout(captured_output), redirect_stderr(captured_output):
                        response = st.session_state.ml_agent.chat(user_input)
                    
                    terminal_output = captured_output.getvalue().strip()
                    if terminal_output:
                        full_response = response + "\n\n**Output:**\n```\n" + terminal_output + "\n```"
                    else:
                        full_response = response
                    
                    # ADD TIMING TO ERROR PATH TOO
                    end_time = time.time()
                    execution_time = end_time - start_time
                    if execution_time < 60:
                        time_str = f"{execution_time:.2f} seconds"
                    else:
                        minutes = int(execution_time // 60)
                        seconds = execution_time % 60
                        time_str = f"{minutes} min {seconds:.1f} sec"
                    
                    full_response += f"\n\n---\n⏱️ **Total processing time: {time_str}**"

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e2:
                    error_msg = f"❌ Error: {str(e2)}"
                    
                    # ADD TIMING TO FINAL ERROR PATH
                    end_time = time.time()
                    execution_time = end_time - start_time
                    if execution_time < 60:
                        time_str = f"{execution_time:.2f} seconds"
                    else:
                        minutes = int(execution_time // 60)
                        seconds = execution_time % 60
                        time_str = f"{minutes} min {seconds:.1f} sec"
                    
                    error_msg += f"\n\n---\n⏱️ **Total processing time: {time_str}**"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            finally:
                # Reset processing flag
                st.session_state.is_processing = False
                time.sleep(0.1)
                st.rerun()

# Show available files for easy reference
if st.session_state.uploaded_files:
    with st.sidebar:
        st.header("📋 Available Files")
        for file_key, file_info in st.session_state.uploaded_files.items():
            st.write(f"• {file_info['name']}")
            if 'shape' in file_info:
                st.write(f"  {file_info['shape'][0]:,} × {file_info['shape'][1]}")

# ========== DOWNLOAD SECTION (ALWAYS VISIBLE) ==========
with st.sidebar:
    st.divider()
    st.header("📥 Download Results")

    temp_dir = tempfile.gettempdir() # Works on all OS
    
    # Trained Model Section
    model_path = os.path.join(temp_dir, "best_model.joblib")

    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        model_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                   time.localtime(os.path.getmtime(model_path)))
        
        with st.container():
            st.write("**🤖 Trained Model**")
            st.caption(f"Size: {model_size:.2f} MB")
            st.caption(f"Updated: {model_time}")
            
            with open(model_path, "rb") as f:
                st.download_button(
                    label="Download Model (.joblib)",
                    data=f,
                    file_name="best_model.joblib",
                    mime="application/octet-stream",
                    key=f"download_model_{int(os.path.getmtime(model_path))}", 
                    width="stretch"
                )
    else:
        st.info("ℹ️ No trained model available")
    
    st.divider()
    
    # Predictions Section
    predictions_path = os.path.join(temp_dir, "predictions.csv")
    
    if os.path.exists(predictions_path):
        try:
            pred_df = pd.read_csv(predictions_path)
            pred_size = os.path.getsize(predictions_path) / 1024  # KB
            pred_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                     time.localtime(os.path.getmtime(predictions_path)))
            
            with st.container():
                st.write("**📊 Predictions**")
                st.caption(f"Rows: {len(pred_df):,} | Size: {pred_size:.1f} KB")
                st.caption(f"Updated: {pred_time}")
                
                # Preview in expander
                with st.expander("🔍 Preview Data", expanded=False):
                    st.dataframe(pred_df.head(10), width = "stretch")
                    
                    # Show prediction distribution
                    if 'predictions' in pred_df.columns:
                        pred_counts = pred_df['predictions'].value_counts()
                        st.write("**Prediction Distribution:**")
                        for pred, count in pred_counts.items():
                            percentage = (count / len(pred_df)) * 100
                            st.write(f"  • {pred}: {count:,} ({percentage:.1f}%)")
                
                # Download button
                csv_data = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions (.csv)",
                    data=csv_data,
                    file_name="predictions.csv",
                    mime="text/csv",
                    key=f"download_predictions_{int(os.path.getmtime(predictions_path))}",
                    width = "stretch"
                )
        except Exception as e:
            st.error(f"❌ Error loading predictions: {str(e)}")
    else:
        st.info("ℹ️ No predictions available")
# ========== END OF DOWNLOAD SECTION ==========

# Cleanup temp files on session end
import atexit

def cleanup_temp_files():
    # Clean up uploaded files
    for file_info in st.session_state.get('uploaded_files', {}).values():
        if 'path' in file_info and os.path.exists(file_info['path']):
            try:
                os.unlink(file_info['path'])
            except:
                pass
    
    # Clean up output files (model and predictions)
    temp_dir = tempfile.gettempdir()
    output_files_to_clean = [
        os.path.join(temp_dir, "best_model.joblib"),
        os.path.join(temp_dir, "predictions.csv")
    ]
    
    for file_path in output_files_to_clean:
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
                print(f"Cleaned up: {file_path}")
            except Exception as e:
                print(f"Could not delete {file_path}: {e}")
    
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except:
        pass

atexit.register(cleanup_temp_files)