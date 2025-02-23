import time

import openai
import pandas
import streamlit as st
import pandas as pd
import json
from io import StringIO
import pyedflib
import matplotlib.pyplot as plt
import re
from muselsl import stream, list_muses, view
from pylsl import StreamInlet
from pylsl.resolve import resolve_stream

from pdf_to_text import pdf_to_text, split_text

# Function to process uploaded files and convert to text
def process_file(file, file_type):
    """Processes the uploaded file and converts it to a textual representation."""
    try:
        if file_type == "txt":
            return file.getvalue().decode("utf-8", errors="replace"), "text"
        elif file_type == "csv":
            df = pd.read_csv(file)
            return df.to_string(index=False), "text"
        elif file_type == "tsv":
            df = pd.read_csv(file, sep="\t")
            return df.to_string(index=False), "text"
        elif file_type == "json":
            data = json.load(file)
            return json.dumps(data, indent=4), "text"
        elif file_type == "edf":
            temp_path = f"/tmp/{file.name}"  # Save the file temporarily
            with open(temp_path, "wb") as temp_file:
                temp_file.write(file.read())

            f = pyedflib.EdfReader(temp_path)
            signal_labels = f.getSignalLabels()
            signals = {label: f.readSignal(idx) for idx, label in enumerate(signal_labels)}
            f.close()

            df = pd.DataFrame(signals)
            return df.to_string(index=False), "text"
    except Exception as e:
        return f"Error processing file: {e}", "error"

# Function to parse plot instructions from LLM responses
def parse_plot_instructions(response, df):
    """Parses LLM response for plot instructions and generates a plot."""
    try:
        # Example instruction: "plot column1 vs column2"
        match = re.search(r"plot (\w+) vs (\w+)", response, re.IGNORECASE)
        if match:
            x_col, y_col = match.groups()
            if x_col in df.columns and y_col in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df[x_col], df[y_col], label=f"{y_col} vs {x_col}")
                ax.set_title(f"{y_col} vs {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.legend()
                st.pyplot(fig)
            else:
                st.error(f"Columns {x_col} or {y_col} not found in the uploaded data.")
    except Exception as e:
        st.error(f"Error interpreting plot instructions: {e}")


def get_muse_data(duration=10):
    """
    Streams Muse data and saves it to a CSV file with channel headers.

    Parameters:
        duration (int): Duration to record data in seconds.
    """
    print("Looking for an EEG stream...")

    # Resolve an EEG stream on the lab network
    streams = resolve_stream('type', 'EEG')
    if not streams:
        print("No EEG stream found. Ensure the Muse device is streaming.")
        return

    # Create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    print("Connected to EEG stream. Recording data...")

    # Get channel names from the stream info
    info = inlet.info()
    description = info.desc()
    channel_names = [channel.child_value("label") for channel in description.child("channels").children()]

    # Print the channel names for reference
    print("Channel Names:", channel_names)

    # Prepare to collect data
    data = []
    start_time = time.time()

    while time.time() - start_time < duration:
        # Get a new sample (data point) from the stream
        sample, timestamp = inlet.pull_sample()
        data.append([timestamp] + sample)

    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=["Timestamp"] + channel_names)

    return df

def run_bot():
    st.title("File Analysis and Conversational Assistant")

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input('Enter your OpenAI API key', type='password')
        if api_key:
            openai.api_key = api_key
            st.success('API key loaded!', icon='✅')
        else:
            st.warning('Please enter your API key!', icon='❌')

        st.subheader('Prompt type')
        prompt_type = st.selectbox('Select Prompt type', ['text', 'image', 'pdf', 'csv', 'EEG', 'robot'])

        # Shared configuration for text, PDF, and CSV
        if prompt_type in ['text', 'pdf', 'csv', 'EEG', 'robot']:
            st.subheader('GPT Version')
            gpt_version = st.selectbox('Select GPT Version', ['gpt-3.5-turbo', 'gpt-4'])
            temperature = st.number_input('Insert Temperature', min_value=0.1, max_value=1.0, value=0.7, step=0.1)
            token_number = st.number_input('Insert Token Max Size', min_value=1, max_value=8192, value=512, step=1)

    # Initialize chat session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_type == 'EEG':
        uploaded_files = st.file_uploader(
            "Upload your EEG files (EDF, JSON, TSV)",
            type=["edf", "json", "tsv"],
            accept_multiple_files=True,
            key="eeg_file_uploader"
        )

        if uploaded_files:
            combined_text = ""  # Combine all file contents into a single text

            for uploaded_file in uploaded_files:
                file_type = uploaded_file.name.split(".")[-1].lower()
                st.write(f"Processing file: {uploaded_file.name}")
                file_content, content_type = process_file(uploaded_file, file_type)

                if content_type == "text":
                    combined_text += f"\n--- File: {uploaded_file.name} ---\n{file_content[:500]}...\n"
                    st.text_area(f"File Content ({uploaded_file.name})", file_content[:1000], height=200, key=f"file_content_{uploaded_file.name}")
                else:
                    st.error(f"Failed to process file: {uploaded_file.name}")

            if combined_text:
                chunks = combined_text  # Limit initial context for large files

                general_prompt = f"Here are the contents of the uploaded files:\n{chunks}\nconsider all of this data and use your knowledge and then inform me in one word(happy, sad, nervous, etc) how was participant feeling (emotional) after listening to this music, and if participant is not happy as an assistance what is your suggestion to make participant happy if you recommended to listen to a music give me some music names."
                # response = st.session_state.conversation.run(general_prompt)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    response = openai.ChatCompletion.create(
                        model=gpt_version,
                        messages=[
                            {"role": "user", "content": general_prompt}
                        ],
                        temperature=temperature,
                        max_tokens=token_number
                    )
                    full_response = response.choices[0].message["content"]
                    message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                    # st.text_area("General Summary", response, height=300, key="general_summary")

    elif prompt_type == 'csv':
        uploaded_csv = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_csv:
            st.write("CSV file uploaded successfully!")
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df)

            if prompt := st.chat_input("Enter your question or analysis request:"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    csv_string = df.to_csv(index=False)
                    analysis_prompt = f"{prompt}\n\nHere is the CSV data for analysis:\n```csv\n{csv_string[:100]}\n```"

                    response = openai.ChatCompletion.create(
                        model=gpt_version,
                        messages=[
                            {"role": "user", "content": analysis_prompt}
                        ],
                        temperature=temperature,
                        max_tokens=token_number
                    )
                    full_response = response.choices[0].message["content"]
                    message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

    elif prompt_type == 'text':
        # Handle text prompt
        if prompt := st.chat_input("Enter your message:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                for response in openai.ChatCompletion.create(
                        model=gpt_version,
                        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                        stream=True,
                        temperature=temperature,
                        max_tokens=token_number
                ):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + " ")
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

    elif prompt_type == 'pdf':
        uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_pdf:
            st.write("PDF file uploaded successfully!")
            extracted_pdf = pdf_to_text(uploaded_pdf, st)

            if prompt := st.chat_input("Enter your question:"):
                st.session_state.messages.append({"role": "user", "content": f"{prompt}\n{extracted_pdf}"})

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""

                    text_segments = split_text(extracted_pdf, max_tokens=token_number)

                    for segment in text_segments:
                        response = openai.ChatCompletion.create(
                            model=gpt_version,
                            messages=[
                                {"role": "user", "content": f"{prompt}\n{segment}"}
                            ],
                            temperature=temperature,
                            max_tokens=token_number
                        )
                        full_response += response.choices[0].message["content"]
                        message_placeholder.markdown(full_response + " ")
                    message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

    elif prompt_type == 'image':
        if prompt := st.chat_input("Enter your image prompt:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                response = openai.Image.create(
                    prompt=prompt,
                    n=1,
                    size="256x256"
                )
                for result in response['data']:
                    full_response += f"![Image]({result['url']})\n"
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

    elif prompt_type == 'robot':
        eeg_data_df = get_muse_data()

        if eeg_data_df is not None:
            csv_string = eeg_data_df.to_csv(index=False)
            general_prompt = f"Here are the EEG data collected by Muse 2:\n{csv_string}\n consider all of this data and according to the previous EEG data tha you analysed, inform me in one word(happy, sad, nervous, etc) how was my feeling (emotional) after listening to this music, and if i am not happy as an assistance what is your suggestion to make me happy if you recommended to listen to a music give me some music names."
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response = openai.ChatCompletion.create(
                    model=gpt_version,
                    messages=[
                        {"role": "user", "content": general_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=token_number
                )
                full_response = response.choices[0].message["content"]
                message_placeholder.markdown(full_response)

if __name__ == "__main__":
    run_bot()
