import boto3
from botocore.exceptions import BotoCoreError, ClientError
from bot_nlp import process_pinged_text
import settings

# Initialize the Lex client with your AWS credentials from settings.py
lex_client = boto3.client(
    'lexv2-runtime',  # Ensure using 'lexv2-runtime' for Lex V2
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)

def process_user_input(user_input):
    # Ensure that processed_input is a string
    print('lex_chat_interface => 1: ',user_input)
    entities = process_pinged_text(user_input)  # This now returns a list of tuples
    processed_input = ' '.join([ent[0] for ent in entities])  # Join entity texts into a string
    print('lex_chat_interface => 2: ',processed_input)

    try:
        # response = lex_client.recognize_text(  # Changed to recognize_text for Lex V2
        #     botId=settings.LEX_BOT_ID,  # Use botId for Lex V2
        #     botAliasId=settings.LEX_BOT_ALIAS_ID,  # Use botAliasId for Lex V2
        #     localeId=settings.LEX_LOCALE_ID,  # Specify the localeId for Lex V2
        #     sessionId='DefaultUser',  # Use sessionId instead of userId for Lex V2
        #     text=processed_input
        # )
        # # tensorflow code comes
        # print('lex_chat_interface => 3: ',response)
        return processed_input
    except (BotoCoreError, ClientError) as error:
        print(error)
        return None

def handle_user_message(user_input):
    lex_response = process_user_input(user_input)
    if lex_response:
        print("Lex response:", lex_response)
        return lex_response  # Return the entire response for now
    else:
        print("No response received from Lex")
        return "No response received from Lex"

# Testing code - this will not be executed when imported as a module
if __name__ == "__main__":
    user_input = "I would like to buy an Apple Macbook"
    response = handle_user_message(user_input)
    print(response)  # Print the response for testing
