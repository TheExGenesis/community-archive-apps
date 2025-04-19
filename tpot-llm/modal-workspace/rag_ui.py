import modal
from .rag import Model, app

web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    [
        "streamlit==1.32.0",
        "numpy",
    ]
)


@app.function(
    image=web_image,
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
)
@modal.asgi_app()
def ui():
    import streamlit as st
    from uuid import uuid4

    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.session_id = str(uuid4())

    st.title("💬 Text RAG Chat")

    # Initialize our Model
    model = Model()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get model response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = model.respond_to_message.remote(
                    st.session_state.session_id, prompt
                )
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Add a sidebar with some info
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            """
        This is a RAG (Retrieval Augmented Generation) chatbot that uses:
        - BAAI/bge-base-en-v1.5 for embeddings
        - Pre-computed embeddings from our corpus
        """
        )

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid4())
            st.rerun()
