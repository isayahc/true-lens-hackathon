# from litellm import completion
import os
from trulens_eval.feedback.provider.litellm import LiteLLM
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from trulens_eval import TruChain, Feedback, Tru #, LiteLLM
from google.cloud import aiplatform
from litellm import completion

#######################################################################
# RESOURCES
#############
# https://github.com/truera/trulens/blob/main/trulens_eval/examples/expositional/models/google_vertex_quickstart.ipynb

#######################################################################


tru = Tru()
tru.reset_database()


aiplatform.init(
    project = "vectara-404518",
    location="us-central1"
)

feedback_model:LiteLLM = LiteLLM(model_engine="chat-bison")

response = completion(model="chat-bison", messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}])



full_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=
        "Provide a helpful response with relevant background information for the following: {prompt}",
        input_variables=["prompt"],
    )
)

chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])

llm:OpenAI = OpenAI(temperature=0.9, max_tokens=128)

chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=True)

relevance = Feedback(feedback_model.relevance_with_cot_reasons).on_input_output()

prompt_input = 'What is a good name for a store that sells colorful socks?'

tru_recorder = TruChain(chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[relevance])

with tru_recorder as recording:
    llm_response = chain(prompt_input)

tru.get_records_and_feedback(app_ids=[])[0]

tru.run_dashboard()

x=0