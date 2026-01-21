# Few Shot Learningによるセンチメント分析
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. examplesの作成
examples = []
for i, row in df_train_sample.iterrows():
    examples.append({
        "text": row["text"],
        "label": "positive" if row["label"] == 1 else "negative"
    })

# 2. Few Shot Promptの作成
example_formatter_template = """
Review: {text}
Sentiment: {label}\n
"""
example_prompt = PromptTemplate(
    input_variables=["text", "label"],
    template=example_formatter_template,
    )

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="次の映画レビューのセンチメント（positive/negative）を判定してください。",
    suffix="Review: {input}\nSentiment:",
    input_variables=["input"],
    example_separator="\n",
)

# 3. chainの作成
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
chain = few_shot_prompt | llm | StrOutputParser()

# 4. 任意のテキストで予測
test_text = "This movie was boring and too long. I couldn't finish it."
print(chain.invoke({"input": test_text}))


# テストデータから任意のテキストでセンチメント予測
for i, row in df_test_sample.iterrows():
    print("Review:", row["text"][:100].replace("\n", " ") + ("..." if len(row["text"]) > 100 else ""))
    print("Ground Truth:", "positive" if row["label"] == 1 else "negative")
    print("Prediction:", chain.invoke({"input": row["text"]}))
    print("-"*40)