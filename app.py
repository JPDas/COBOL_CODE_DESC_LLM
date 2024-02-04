import streamlit as st
from ctransformers import AutoModelForCausalLM, AutoConfig
import os

class ModelWrapper:
    def __init__(self, 
                 model_selected,
                 no_words, 
                 desc_style,
                 few_shot) -> None:

        if model_selected == 'Llama2':
            self.model_name = "Models/llama-2-7b-chat.ggmlv3.q8_0.bin"
            self.model_type = 'llama'
        elif model_selected == 'Mistral':
            self.model_name = "Models/mistral-7b-instruct-v0.2.Q8_0.gguf"
            self.model_type = 'mistral'
        elif model_selected == 'starCoder':
            self.model_name = "Models/llama-2-7b-chat.ggmlv3.q8_0.bin"
            self.model_type = 'llama'
        else:
            self.model_name = "Models/llama-2-7b-chat.ggmlv3.q8_0.bin"
            self.model_type = 'llama'
        
        self.no_words = no_words
        self.desc_style = desc_style
        self.fewshot_enabled = few_shot
        self.llm_model = None
    
    def create_llm_model(self):
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            model_type=self.model_type
        )            
    def build_prompt(self, input_code, chunking_code = False):
        if chunking_code:
            template = f''' 
            In continuation of previous tasks, please generate the description of below cobol code for {self.desc_style} in {self.no_words} number of words.
            
            {input_code}
            '''
        else:
            template = f''' 
            You are expert in Cobol code and your task is to generate the concise description from below cobol code for {self.desc_style} in {self.no_words} number of words.
            
            {input_code}
            '''
        
        return template
    def generate_response(self, input_code, chunking_code = False):

        prompt = self.build_prompt(input_code, chunking_code=chunking_code)
        
        print(prompt)

        response = self.llm_model(prompt, temperature=0.1, max_new_tokens=100)

        return response
    
def file_selector(folder_path='.', file_desc = 'Select a file'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox(file_desc, filenames)
    return os.path.join(folder_path, selected_filename)

if __name__ == '__main__':
    st.set_page_config(page_title="Cobol Code Descriptions",
                       page_icon="cobol.png",
                       layout="centered",
                       initial_sidebar_state="collapsed")
    
    st.image("cobol.png", width = 250)
    st.header("Generate Descriptions")

    filename = file_selector(folder_path="Inputs", file_desc="Select Cobol file")
    st.write('You selected `%s`' % filename)

    st.subheader('Models and parameters')

    model_selected = st.selectbox("Select Model ", 
                                  ('Llama2', 'Mistral', 'starCoder'),
                                  index=0)
    col1, col2 = st.columns([5,5])

    with col1:
        no_words = st.slider('No of Words', 256, 2048, 256)
    
    with col2:
        desc_style = st.selectbox("Writing descriptions for ", 
                                  ('Developer', 'Business People'),
                                  index=0)
    
    col3, col4 = st.columns([5,5])
    with col3:
        temperature = st.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    with col4:
        top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        
    few_shot = st.checkbox('Few-Short')
    submit = st.button("Generate",)


    if submit:
        print('Calling LLM functionality')

        file = open(filename,"r")
        input_code = file.read()


        llm = ModelWrapper(model_selected,
                           no_words,
                           desc_style,
                           few_shot)
        
        llm.create_llm_model()

        response = llm.generate_response(input_code)

        print(response)
        st.subheader('Description')
        st.write(response)

        file.close()




        
    
