import openai
import os


class ChatGPT:
    def __init__(self):
        # Load your API key from an environment variable or secret management service
        api_key = os.getenv("OPENAI_API_KEY")

        # Initialize the OpenAI API client
        openai.api_key = api_key

    def query_chat_gpt(self, query, num_results=1, **kwargs):
        retry = 3
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",  # "gpt-3.5-turbo",  # $0.002 / 1k
                    # model="gpt-4",  # $0.03 / 1k
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"{query}"},
                    ],
                    n=num_results,
                    **kwargs
                )
                if num_results == 1:
                    return response['choices'][0]['message']['content']
                else:
                    rets = []
                    for i in range(num_results):
                        ret = response['choices'][i]['message']['content']
                        rets.append(ret)
                    return rets
            except openai.error.APIError:
                retry -= 1
                if retry == 0:
                    raise

    async def query_chat_gpt_async(self, query, model="gpt-3.5-turbo-16k", **kwargs):
        response = await openai.ChatCompletion.acreate(
            model=model,  # "gpt-3.5-turbo",  # $0.002 / 1k
            # model="gpt-4",  # $0.03 / 1k
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{query}"},
            ],
            n=1,
            **kwargs
        )

        ret = response['choices'][0]['message']['content']
        return ret

    def prompt_to_extract_code_block(self, english_with_code):
        prefix = """
Your job is to separate the English description and the code. Wrap code in code block. Only answer the code block. No explanation is needed. If the input has only code and no english descriptions, return the input as is. It is possible that the code is a part of the English sentence.
##### input starts #####
        """
        postfix = """
##### input ends #####
"""
        output = f"{prefix}" \
                 f"{english_with_code}" \
                 f"{postfix}"
        return output

    def extract_code_block(self, english_with_code):
        query = self.prompt_to_extract_code_block(english_with_code)
        output = self.query_chat_gpt(query)
        output = output.strip('`')
        output = output.strip()
        return output


def list_models():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print(openai.Model.list())


def synthesize_rel():
    chatgpt = ChatGPT()
    ans = chatgpt.synthesize_rel_from_nl("hello what's up chatgpt?")
    print(ans)


def chat():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # $0.002 / 1k
        # model="gpt-4",  # $0.03 / 1k
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is REL?"},
        ]
    )
    ret = response['choices'][0]['message']['content']
    print(ret)




if __name__ == "__main__":
    chat()
