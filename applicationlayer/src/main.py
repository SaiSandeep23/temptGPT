from flask import Flask, request, jsonify
from DialogueManager import DialogueManager
from StateManager import StateManager

app = Flask(__name__)
state_manager = StateManager()
    
dialogue_manager = DialogueManager(state_manager)

@app.route('/message', methods=['POST'])
def message():
    data = request.json
    session_id = data['sessionId']
    user_input = data['message']
    
    print('user_input',user_input)
    print('session_id',session_id)
    
    response = dialogue_manager.handle_message(session_id, user_input)
    print('___response___',response,'____')
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5000)
