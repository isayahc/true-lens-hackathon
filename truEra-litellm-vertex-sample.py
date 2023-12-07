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


tru = Tru()
tru.reset_database()

# LiteLLM.vertex_project = "vectara-404518"
# LiteLLM.vertex_location = "us-central1"

aiplatform.init(
    project = "vectara-404518",
    location="us-central1"
)

litellm = LiteLLM(model_engine="chat-bison")

response = completion(model="chat-bison", messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}])




# os.environ["VERTEX_PROJECT"] = "vectara-404518"
# os.environ["VERTEX_LOCATION"] = "us-central1"

# sample_endpoint = '//aiplatform.googleapis.com/projects/vectara-404518/locations/us-central1/publishers/google/models/chat-bison@002'



# litellm = LiteLLM(model_engine="chat-bison",endpoint=sample_endpoint) # choose any model here!

# LiteLLM.endpoint = sample_endpoint
# LiteLLM.vertex_project = "vectara-404518"
# LiteLLM.vertex_location = "us-central1"


full_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=
        "Provide a helpful response with relevant background information for the following: {prompt}",
        input_variables=["prompt"],
    )
)

chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])

llm = OpenAI(temperature=0.9, max_tokens=128)

chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=True)

relevance = Feedback(litellm.relevance_with_cot_reasons).on_input_output()

prompt_input = 'What is a good name for a store that sells colorful socks?'

tru_recorder = TruChain(chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[relevance])

with tru_recorder as recording:
    llm_response = chain(prompt_input)

tru.get_records_and_feedback(app_ids=[])[0]

tru.run_dashboard()

x=0