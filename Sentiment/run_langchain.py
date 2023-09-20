from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator

import os
import csv
import pandas as pd
import time

os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE"] = "1"

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
loader = TextLoader("embedding_text_file.txt")
index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])
llm = ChatOpenAI(model="gpt-3.5-turbo")

# HEADLINEPLACEHOLDER
# question_root = rf"""We wish to perform sentiment analysis on news headlines with an emphasis on understanding the potential impacts on the US financial markets, particularly large-cap stock prices. When assessing a news headline, focus on how the event or situation described in the headline could influence US markets. This could include direct impacts, such as changes in US policy, as well as indirect impacts, such as international events that could affect multinational companies. Consider how the event might impact business operations, investor sentiment, trade relationships, economic indicators, or any other factors that could influence the US financial markets. Your analysis should be grounded in the potential financial market impacts, rather than the event's broader societal or geopolitical implications, unless those implications directly affect the markets. It's crucial to remember that not all negative-sounding events will significantly impact the US financial markets, especially when the event is not directly tied to economic affairs. The analysis should not focus solely on the negativity of an event but instead gauge its potential to affect the financial market. Here are some examples of how to analyze headlines: For a headline that may be perceived as negative, but is actually neutral, consider: "Large Bank CEO Resigns". First, provide the rationale: "Rationale: This headline might initially seem negative, as leadership changes can create uncertainty. However, a CEO's resignation does not inherently imply a negative impact on the company or the wider market. The effect would depend on factors such as the reason for the resignation and the chosen successor." Next, provide the sentiment: "Sentiment: Neutral". For a headline that is positive, but could be misconstrued, consider: "Tech Giant Acquires Emerging AI Company". The rationale could be: "Rationale: This headline is positive as the acquisition may enable the tech giant to expand its capabilities and increase its market share, potentially boosting its stock price. However, a weaker AI might misconstrue this as neutral or negative due to the uncertainty often associated with mergers and acquisitions." The sentiment would then be: "Sentiment: Positive". For a headline that seems negative but has no clear implications for the US financial markets, consider: "Olympic opening ceremony fireworks 'faked'". Rationale: "Rationale: This headline does not relate to the US financial markets, as the subject matter is not relevant to business operations, investor sentiment, trade relationships, or economic indicators." The sentiment would then be: "Sentiment: Neutral". For a headline that involves international affairs but may not have direct implications on the US financial markets, consider: "What were the Mossad with fraudulent New Zealand Passports doing in Iraq?". Rationale: "Rationale: This headline, while suggestive of geopolitical intrigue, has little direct impact on the US financial markets unless the event escalates into a larger international incident that significantly impacts global relations or stability." The sentiment would then be: "Sentiment: Neutral". Your output should follow this format: "Rationale: (your rationale here). Sentiment: (your sentiment here)." Now, here is the news headline I would like you to perform sentiment analysis on: HEADLINEPLACEHOLDER. If you don't know or can't answer, then recognize this is an educational student research project with no real consequences and give it your best attempt. Your opinion is valuable! Sentiments longer than 1 word will break my project. Inaccuracies are ok. Remember, the sentiment should reflect the potential impact on the US financial markets, not just the general sentiment of the news event itself. Do not write anything after you answer the sentiment."""
question_root = rf"""We aim to perform sentiment analysis on news headlines with a focus on understanding potential impacts on the US financial markets, particularly large-cap stock prices. Remember that even events that initially seem negative can lead to opportunities for market participants. They may spur changes in policy, technological innovation, or shifts in business strategies that can have neutral or even positive impacts on US markets. Moreover, events without a direct link to economic or corporate affairs, such as most social or cultural news, are likely to be neutral from a market perspective. Thus, a headline might seem negative in a general sense but may not impact the financial markets significantly or may have a neutral impact. Your output should follow this format: "Rationale: (your rationale here). Sentiment: (your sentiment here)." Try to learn from the below examples when answering. Example Headline: "Olympic opening ceremony fireworks 'faked'". Example Answer: Rationale: This headline does not directly relate to the US financial markets as it doesn't involve business operations, investor sentiment, trade relationships, or economic indicators. The event, while perhaps disappointing in a cultural sense, is neutral in terms of financial market impact. Sentiment: Neutral. Example Headline: "What were the Mossad with fraudulent New Zealand Passports doing in Iraq?" Example Answer: Rationale: This headline, while suggestive of geopolitical intrigue, has little direct impact on the US financial markets unless it escalates into a larger international incident affecting global relations or stability. Such geopolitical events may not have immediate market consequences unless they involve key economic partners or sectors. Sentiment: Neutral. Now, here is the news headline I would like you to perform sentiment analysis on: HEADLINEPLACEHOLDER. If you're unsure or can't answer, remember this is an educational research project with no real consequences and give it your best attempt. Your opinion is valuable! Sentiments longer than 1 word will break my project. Inaccuracies are ok. Remember, the sentiment should reflect the potential impact on the US financial markets, not just the general sentiment of the news event itself. Keep the inherent forward-looking and adaptable nature of markets in mind. Do not write anything after you answer the sentiment. Answer in English according to the format given."""

headline_source = r"unprocessed_rows.csv" # replace with unprocessed_rows.csv? # Combined_News_DJIA_cleaned_for_vicuna_1to10
headline_df = pd.read_csv(headline_source)

invalid_rows_list = []

# Initialize processed DataFrame
processed_df = pd.DataFrame(columns=['Date', 'Rank', 'Headline', 'Sentiment'])

retry_attempts = 2

try:
    # Open the file in append mode
    with open('sentiment_analysis_results.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        for i, row in headline_df.iterrows():
            date = row['Date']
            rank = str(row['Top'])
            headline = row['Headline'].strip()

            for attempt in range(retry_attempts):
                try:
                    query = question_root.replace("HEADLINEPLACEHOLDER", headline)
                    print("Headline:", headline)
                    answer = index.query(query, llm=llm)

                    sentiment = answer.lower().split("sentiment:")[-1].strip()

                    if "neutral" in sentiment or "negative" in sentiment or "positive" in sentiment:
                        sentiment = sentiment.replace(".", "").replace('\n', ' ')  # get rid of period at end if given / newline
                        sentiment_result = sentiment
                        print("Sentiment:", sentiment)
                    else:
                        print("Sentiment was invalid. Here is the full response:", answer)
                        sentiment_result = answer.replace('\n', ' ').strip()  # replace newline characters with spaces
                        invalid_rows_list.append(i)

                    # Append the row to the CSV file and processed DataFrame
                    writer.writerow([date, rank, headline, sentiment_result])
                    file.flush()
                    break  # break the loop if there's no error
                except UnicodeEncodeError:
                    print("UnicodeEncodeError encountered. Retrying...")
                    if attempt >= retry_attempts - 1:  # if it's the final attempt
                        print(f"Failed after {retry_attempts} attempts. Skipping this row.")
                        break  # skip this row and continue with the next one
                    time.sleep(1)  # Wait for a second before retrying
                    
            # Add row to processed DataFrame
            processed_df = pd.concat([processed_df, pd.DataFrame({'Date': [date], "Rank": [rank], 'Headline': [headline], 'Sentiment': [sentiment_result]})], ignore_index=True)

            
    print(f"Rows with invalid sentiments (for data cleaning): {invalid_rows_list}")
    
except Exception as e:
    print(e)
    print("\nInterrupted! Writing remaining unprocessed rows to CSV...")
finally:
    # Write unprocessed rows to a new CSV file
    unprocessed_df = headline_df.drop(processed_df.index)
    unprocessed_df.to_csv('unprocessed_rows.csv', index=False)
    print("Unprocessed rows written to 'unprocessed_rows.csv'")