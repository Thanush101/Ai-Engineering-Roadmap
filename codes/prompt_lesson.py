from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate , PromptTemplate
from dotenv import load_dotenv

load_dotenv()


llm = init_chat_model("groq:llama-3.3-70b-versatile",temperature=1,max_tokens=1000,max_retries=3)

zero_shot_promot  =ChatPromptTemplate.from_messages([
    (
        "system",""" You are influencer with 10 years of experience craft a social media post that will get the most engagement.
        dont give any markdown or emojis. 
        
        Provide a 500 words content for a post that will get the most engagement in the specific topis user provides

        **IMPORTANT**
        Provide Clickbait title
        Provide 100 words post content Dont give any additional information
        Contact details no. 122342113 and email id is 123123123@123.com
        The contact details should be in the contact section only
        The hashtags should be in the hashtags section only not in any other section 
        Dont provide the hashtag in any other section or field

        ***VERY IMPORTANT***
        In the tweet Dont provide any hashtag or contact details


        IMPORTANT: **STRICT JSON OUTPUT FORMAT:**
        {{
            "post": [
                {{
      "platform": "Instagram",
      "title": "",
      "content": "",
      "poll" : " Option one , Option two",
      "hashtags": ["#NewFeature", "#LaunchDay"]
        }},

             {{
      "platform": "Facebook",
      "title": "",
      "content": "",
      "contact" : "",
      "hashtags": ["#NewFeature", "#LaunchDay"]
        }},
             {{
      "platform": "Twitter",
      "Tweet": "",
      "hashtags": ["#NewFeature", "#LaunchDay"]
        }}
         
            """,
        
    ),
    (
        "human", "This is the topic :{topic} "

    )

])



format_prompt  = ChatPromptTemplate.from_messages([
    ("system", """You are Social media formator with 10 years of experience craft a social media post in specific format only 
     

     Content : {content}


     **IMPORTANT**
        Provide Clickbait title
        Provide 100 words post content Dont give any additional information
        Contact details no. 122342113 and email id is 123123123@123.com
        The contact details should be in the contact section only
        The hashtags should be in the hashtags section only not in any other section 
        Dont provide the hashtag in any other section or field
     
     


       IMPORTANT: **STRICT JSON OUTPUT FORMAT:**
        {{
            "post": [
                {{
      "platform": "Instagram",
      "title": "",
      "content": "",
      "poll" : " Option one , Option two",
      "hashtags": ["#NewFeature", "#LaunchDay"]
        }},

             {{
      "platform": "Facebook",
      "title": "",
      "content": "",
      "contact" : "",
      "hashtags": ["#NewFeature", "#LaunchDay"]
        }},
             {{
      "platform": "Twitter",
      "Tweet": "",
      "hashtags": ["#NewFeature", "#LaunchDay"]
        }}


     
     
     """)

])


cot_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        """You are a helpful assistant that solves problems step by step.
        For each problem, follow this format:
        1. Understanding: Restate what the problem is asking
        2. Planning: Outline your approach to solve it
        3. Step-by-step solution: Work through each step clearly
        4. Final answer: Provide the final result

        Think through each step carefully and show your reasoning.
        
"""
    ),
    (
        "human", 
        "Problem: {problem}\n\nSolve this step by step:"
    )
])

# cot_prompt2= PromptTemplate(
#     input_variables=[],
#     template= """
#     You are 

#      """
# )

# chain = zero_shot_promot | llm 
# chain2 = format_prompt| llm
# cot_chain = cot_prompt | llm

# rs = cot_chain.invoke("problem: A complex algorithm outputs one of five results. Which input is most likely if the output is X?,How to Answer: Instead of forward simulating all inputs, test each output option against known inputs to find the matching case quickly.")
# content = rs.content
# print(content)

