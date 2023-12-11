import requests
import json
import settings

def lambda_handler(event, context):
    ngrok_url = settings.processChatEndPoint
    print("Received inputTranscript:", event["inputTranscript"], event["sessionId"])
    response = requests.post(ngrok_url, json={"message": event['inputTranscript'],"sessionId": event["sessionId"]})

    # Try to parse the response as JSON
    try:
        response_content = response.json()
        # Extract the message from the ‘response’ key
        print("response_content", response.content)
        actual_message = response_content.get('response', "Default response")
        print('response_content', actual_message)
    except json.JSONDecodeError:
        # Fallback to plain text if JSON parsing fails
        actual_message = response.text

    print('again', actual_message)
    # Build the response for Lex
    lex_response = {
        "messages": [{
            "contentType": "PlainText",
            "content": actual_message
        }],
        "sessionState": {
            "dialogAction": {
                "type": "Close"
            },
            "intent": {
                "name": event['sessionState']['intent']['name'],
                "state": "Fulfilled"
            }
        }
    }
    
    return lex_response
