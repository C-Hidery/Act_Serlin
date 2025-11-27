# Act_Client.py - Serlin Transformer Client
from re import S
import socket
import json
import datetime
from urllib import response

class SerlinClient:
    
    def __init__(self, server_host, server_port):
        self.server_host = server_host
        self.server_port = server_port
        self.user_id = "default_user"
        self.show_thinking = True
        self.timeout = 1000  # Default timeout in seconds
        self.debug_msg = False
        self.server_opened = False
        print(f"Try to connect to {server_host}:{server_port}...")
        gol = 0
        while gol < 3:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.settimeout(self.timeout)
                self.client_socket.connect((self.server_host, self.server_port))
                self.server_opened = True
                break
            except Exception as e:
                gol += 1
                print(f"Connection attempt {gol} failed: {e}")
                if gol == 3:
                    print("Failed to connect to server after 3 attempts.")
    def send_msg(self, command, data=None):
        if data is None:
            data = {}
        
        request = {
            "user_id": self.user_id,
            "command": command,
            "data": data
        }
        try:
            # Send request
            request_json = json.dumps(request).encode('utf-8')
            # Send message length
            request_length = len(request_json)
            self.client_socket.send(request_length.to_bytes(4, byteorder='big'))
            # Send message data
            self.client_socket.send(request_json)
            if self.debug_msg: print(f"Request sent ({request_length} bytes)")
            
        except socket.timeout:
            return {"status": "error", "message": "Connection timeout"}
        except ConnectionRefusedError:
            return {"status": "error", "message": "Connection refused - server may not be running"}
        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}
    
    def recv_msg(self):
        try:
            # Receive response length
            length_data = self.client_socket.recv(4)
            if not length_data:
                self.client_socket.close()
                return {"status": "error", "message": "No response length received"}
            
            response_length = int.from_bytes(length_data, byteorder='big')
            if self.debug_msg: print(f"Expecting response of {response_length} bytes")
        
            # Receive response data
            response_data = b""
            bytes_received = 0
            while bytes_received < response_length:
                chunk = self.client_socket.recv(min(4096, response_length - bytes_received))
                if not chunk:
                    break
                response_data += chunk
                bytes_received += len(chunk)
                if self.debug_msg: print(f"Received {bytes_received}/{response_length} bytes")
        
            #client_socket.close()
        
            if bytes_received < response_length:
                return {"status": "error", "message": f"Incomplete response: {bytes_received}/{response_length} bytes"}
        
            # Parse response
            response = json.loads(response_data.decode('utf-8'))
            print("Response received and parsed successfully")
            return response
        except socket.timeout:
            return {"status": "error", "message": "Connection timeout"}
        except ConnectionRefusedError:
            return {"status": "error", "message": "Connection refused - server may not be running"}
        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}
    def send_and_recv(self,command,data=None):
        if data is None:
            data = {}
        self.send_msg(command,data)
        response = self.recv_msg()
        if response.get('status') == 'success':
            return response
        else:
            print(f"Error: {response.get('message', 'Unknown error')}")
    def chat(self, user_input):
        """Send chat message to server"""
        response = self.send_and_recv('chat', {
            'user_input': user_input,
            'show_thinking': self.show_thinking
        })
        return response['data']['response']
    def get_system_status(self):
        """Get system status from server"""
        response = self.send_and_recv('get_system_status')
        return response
    
    def get_conversation_summary(self):
        """Get conversation summary from server"""
        response = self.send_and_recv('get_conversation_summary')
        return response
    
    def add_training_data(self, questions, answers):
        """Add training data to server"""
        response = self.send_and_recv('add_training_data', {
            'questions': questions,
            'answers': answers
        })
        return response
    
    def train_model(self, epochs=100, batch_size=4, learning_rate=0.0001, early_stopping=True):
        """Train model on server"""
        response = self.send_and_recv('train', {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'early_stopping': early_stopping
        })
        return response
    
    def validate_training_data(self):
        """Validate training data on server"""
        response = self.send_and_recv('validate_training_data')
        return response
    
    def get_vocabulary_info(self):
        """Get vocabulary information from server"""
        response = self.send_and_recv('get_vocabulary_info')
        return response
    
    def expand_vocabulary(self, words_text):
        """Expand vocabulary on server"""
        response = self.send_and_recv('expand_vocabulary', {
            'words_text': words_text
        })
        return response
    
    def reload_vocabulary(self):
        """Reload vocabulary on server"""
        response = self.send_and_recv('reload_vocabulary')
        return response
    
    def fix_model_vocab(self):
        """Fix model vocabulary synchronization on server"""
        response = self.send_and_recv('fix_model_vocab')
        return response
    
    def reset_model(self):
        """Reset model on server"""
        response = self.send_and_recv('reset_model')
        return response
    
    def export_conversation(self, filename=None):
        """Export conversation from server"""
        response = self.send_and_recv('export_conversation', {
            'filename': filename
        })
        return response
    
    def load_model(self, path=None):
        """Load model on server"""
        response = self.send_and_recv('load_model', {
            'path': path
        })
        return response
    
    def save_model(self, path=None):
        """Save model on server"""
        response = self.send_and_recv('save_model', {
            'path': path
        })
        return response
    
    def create_training_template(self):
        """Create training template on server"""
        response = self.send_and_recv('create_training_template')
        fn = f"training_example_{datetime.datetime}.json"
        with open(fn,'w',encoding='utf-8') as f:
            json.dump(response.get('data'), f, ensure_ascii=False, indent=2)
        print(f"Training data template created: {fn}")
        return response
    
    def batch_train_from_json(self, file_path):
        """Batch train from JSON file on server"""
        with open(file_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
        response = self.send_and_recv('batch_train_from_json',{
                'training_data': training_data
            })
        return response

def print_help():
    print("Available commands:")
    print("  'exit'/'quit' - End conversation")
    print("  'status' - View system status")
    print("  'summary' - Show conversation summary")
    print("  'export' - Export conversation history to file")
    print("  'train' - Enter training mode")
    print("  'training options' - Show training options menu")
    print("  'training status' - Show training status")
    print("  'load model' - Load model from file")
    print("  'help' - Show this help information")
    print("  'silent' - Toggle thinking process display (on/off)")
    print("  'vocabulary' - View vocabulary")
    print("  'expand vocab' - Add vocabulary")
    print("  'reload vocab' - Reload vocabulary")
    print("  'fix model' - Manually sync vocabulary")
    print("  'validate data' - Validate training data")
    print("  'reset model' - Reset model")
    print("  'create template' - Create training data template")
    print("  'set parameters' - Set AI parameters")
    print("  'set timeout' - Set connection timeout")
    print("  'debug_msg` - Show debug message")
    print("-" * 50)

def interactive_training_mode(client):
    """Interactive training mode"""
    print("\n=== Serlin Transformer Interactive Training Mode ===")
    print("Input training data in format:")
    print("  Question: your question")
    print("  Expected reply: expected answer") 
    print("Input 'done' to finish data input")
    print("Input 'train' to start training")
    print("Input 'exit' to return to main menu")
    print("=" * 50)
    
    questions = []
    answers = []
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                return False
            elif user_input.lower() in ['done', 'finish']:
                break
            elif user_input.lower() in ['train', 'start']:
                if questions and answers:
                    print(f"Ready to train, total {len(questions)} Q&A pairs")
                    
                    # Add training data
                    add_result = client.add_training_data(questions, answers)
                    if add_result: print(f"Successfully added {len(questions)} training pairs")
                    else: 
                        print("Failed to add training data")
                        continue
                    # Ask for training parameters
                    print("\n=== Training Parameter Settings ===")
                    try:
                        epochs = int(input("Training epochs (default 50): ") or "50")
                        batch_size = int(input("Batch size (default 2): ") or "2")
                        learning_rate = float(input("Learning rate (default 0.0001): ") or "0.0001")
                        early_stopping = input("Enable early stopping? (y/N): ").strip().lower() == 'y'
                    except ValueError:
                        print("Invalid input, using default parameters")
                        epochs = 50
                        batch_size = 2
                        learning_rate = 0.0001
                        early_stopping = True
                    
                    print(f"Starting training: {epochs} epochs, batch size: {batch_size}, learning rate: {learning_rate}")
                    
                    # Start training
                    train_result = client.train_model(
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        early_stopping=early_stopping
                    )
                    
                    if train_result:
                        print("Training completed successfully!")
                        if 'final_loss' in train_result:
                            print(f"Final loss: {train_result['final_loss']:.4f}")
                    else:
                        print(f"Training failed: {train_result.get('message')}")
                
                    questions = []
                    answers = []
                else:
                    print("No training data, please add data first!")
                continue
            
            # Parse training data
            if user_input.startswith("Question:"):
                question = user_input[9:].strip()
                questions.append(question)
                print(f"Recorded question: {question}")
            elif user_input.startswith("Expected reply:"):
                answer = user_input[15:].strip()
                answers.append(answer)
                print(f"Recorded expected reply: {answer}")
            else:
                print("Format error! Please use:")
                print("  Question: your question")
                print("  Expected reply: expected answer")
    
        except Exception as e:
            print(f"Error processing input: {e}")
            continue
    
    return True

def show_training_options(client):
    """Show training options menu"""
    print("\n=== Training Options ===")
    print("1. Interactive training - manually input training data")
    print("2. JSON file batch training - load from JSON file")
    print("3. Create training template - generate JSON template file")
    print("4. Return to main menu")
    
    choice = input("Please select training method (1-4): ").strip()
    
    if choice == '1':
        interactive_training_mode(client)
    elif choice == '2':
        file_path = input("Please enter JSON training file path: ").strip()
        if file_path:
            result = client.batch_train_from_json(file_path)
            if result:
                print(f"Success: {result.get('message')}")
                # Ask if user wants to train now
                train_now = input("Start training now? (y/N): ").strip().lower()
                if train_now == 'y':
                    # Ask for training parameters
                    print("\n=== Training Parameter Settings ===")
                    try:
                        epochs = int(input("Training epochs (default 100): ") or "100")
                        batch_size = int(input("Batch size (default 4): ") or "4")
                        learning_rate = float(input("Learning rate (default 0.0001): ") or "0.0001")
                        early_stopping = input("Enable early stopping? (y/N): ").strip().lower() == 'y'
                    except ValueError:
                        print("Invalid input, using default parameters")
                        epochs = 100
                        batch_size = 4
                        learning_rate = 0.0001
                        early_stopping = True
                    
                    train_result = client.train_model(
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        early_stopping=early_stopping
                    )
                    
                    if train_result:
                        print("Training completed successfully!")
                    else:
                        print(f"Training failed: {train_result.get('message')}")
            else:
                print(f"Error: {result.get('message')}")
    elif choice == '3':
        file_name = input("Please enter template file name (default training_template.json): ").strip() or "training_template.json"
        result = client.create_training_template(file_name)
        if result:
            print(f"Template created: {result.get('data', {}).get('file_path')}")
        else:
            print(f"Error creating template: {result.get('message')}")
    elif choice == '4':
        return
    else:
        print("Invalid selection")

def show_training_status(client):
    """Show training status"""
    status_result = client.get_system_status()
    if status_result:
        data = status_result.get('data', {})
        print(f"\nTraining data count: {data.get('training_data_count', 0)}")
        print(f"Vocabulary size: {data.get('vocabulary_size', 0)}")
    else:
        print(f"Error getting status: {status_result.get('message')}")

def main():
    # Get server connection details
    server_host = input("Enter server host (default: localhost): ").strip() or 'localhost'
    try:
        server_port = int(input("Enter server port (default: 5555): ").strip() or '5555')
    except ValueError:
        server_port = 5555
        print("Invalid port, using default: 5555")
    
    # Initialize client
    client = SerlinClient(server_host, server_port)
    print(f"{client.server_opened}")
    if not client.server_opened:
        print("Unable to connect to server, exiting...")
        return
    # Test connection
    print("Connecting to server...")
    test_result = client.get_system_status()
    if not test_result:
        print(f"Failed to connect to server: {test_result.get('message')}")
        return
    
    print("Connected to Serlin Server successfully!")
    
    # Get user ID
    client.user_id = input("Please enter user ID: ").strip() or "default_user"
    
    print(f"\nWelcome, user {client.user_id}!")
    print_help()
    
    while True:
        try:
            user_input = input(f"{client.user_id}: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                # Show summary before exiting
                summary_result = client.get_conversation_summary()
                if summary_result:
                    print(f"\n{summary_result.get('data', {}).get('summary', 'No summary available')}")
                export_choice = input("Export conversation history? (y/N): ").strip().lower()
                if export_choice == 'y':
                    export_result = client.export_conversation()
                    if export_result:
                        print(f"Conversation exported to: {export_result.get('data', {}).get('filename')}")
                    else:
                        print(f"Export failed: {export_result.get('message')}")
                print("Thank you for using Serlin, goodbye!")
                break
                
            elif user_input.lower() in ['status']:
                status_result = client.get_system_status()
                if status_result:
                    data = status_result.get('data', {})
                    print(f"\n=== System Status ===")
                    print(f"User ID: {data.get('user_id')}")
                    print(f"Conversation history records: {data.get('conversation_history_count', 0)}")
                    print(f"Current topics: {', '.join(data.get('current_topics', []))}")
                    print(f"Training data count: {data.get('training_data_count', 0)}")
                    print(f"Vocabulary size: {data.get('vocabulary_size', 0)}")
                    print(f"Model vocabulary size: {data.get('model_vocabulary_size', 0)}")
                    
                    profile = data.get('personality_profile', {})
                    if profile:
                        print(f"Personality profile:")
                        print(f"  Formality level: {profile.get('formality', 0):.2f}")
                        print(f"  Humor level: {profile.get('humor_level', 0):.2f}")
                        print(f"  Detail level: {profile.get('detail_level', 0):.2f}")
                        print(f"  Empathy level: {profile.get('empathy_level', 0):.2f}")
                        print(f"  Curiosity level: {profile.get('curiosity_level', 0):.2f}")
                    print("================")
                else:
                    print(f"Error getting status: {status_result.get('message')}")
                continue
                
            elif user_input.lower() in ['summary']:
                summary_result = client.get_conversation_summary()
                if summary_result:
                    print(f"\n{summary_result.get('data', {}).get('summary', 'No summary available')}")
                else:
                    print(f"Error getting summary: {summary_result.get('message')}")
                continue
                
            elif user_input.lower() in ['vocabulary', 'vocab']:
                vocab_result = client.get_vocabulary_info()
                if vocab_result:
                    print(f"\nVocabulary size: {vocab_result.get('vocab_size', 0)}")
                    print(f"Model vocabulary size: {vocab_result.get('model_vocab_size', 0)}")
                    print(f"Sample vocabulary: {vocab_result.get('sample_words', [])}")
                else:
                    print(f"Error getting vocabulary info: {vocab_result.get('message')}")
                continue

            elif user_input.lower() in ['expand vocab', 'expand vocabulary']:
                words_to_add = input("Please enter words to add (separated by spaces): ").strip()
                if words_to_add:
                    result = client.expand_vocabulary(words_to_add)
                    if result:
                        print(f"Successfully expanded vocabulary, new size: {result.get('data', {}).get('new_vocab_size')}")
                    else:
                        print(f"Error expanding vocabulary: {result.get('message')}")
                continue
            elif user_input.lower() in ['set timeout']:
                r1 = input("Please enter connection timeout in seconds (default 1000): ").strip()
                if r1 == 'None':
                    client.timeout = None
                else:
                    client.timeout = int(r1) if r1.isdigit() else 1000
                print(f"Connection timeout set to {client.timeout} seconds")
                continue
            elif user_input.lower() in ['validate data', 'validate']:
                result = client.validate_training_data()
                if result:
                    data = result.get('data', {})
                    print(f"Validation result: {data.get('valid_count', 0)}/{data.get('total_count', 0)} valid data")
                    if data.get('is_valid'):
                        print("Training data validation passed")
                    else:
                        print("Training data validation failed")
                else:
                    print(f"Error validating data: {result.get('message')}")
                continue
                
            elif user_input.lower() in ['fix model', 'fix']:
                print("Executing model repair...")
                result = client.fix_model_vocab()
                if result:
                    print("Repair completed successfully")
                else:
                    print(f"Repair failed: {result.get('message')}")
                continue
                
            elif user_input.lower() in ['reload vocab', 'reload vocabulary']:
                result = client.reload_vocabulary()
                if result:
                    print("Vocabulary reloaded successfully")
                else:
                    print(f"Error reloading vocabulary: {result.get('message')}")
                continue
                
            elif user_input.lower() in ['reset model', 'reset']:
                result = client.reset_model()
                if result:
                    print("Model reset successfully")
                else:
                    print(f"Error resetting model: {result.get('message')}")
                continue
                
            elif user_input.lower() in ['export']:
                filename = input("Enter filename (optional): ").strip() or None
                result = client.export_conversation(filename)
                if result:
                    print(f"Conversation exported to: {result.get('data', {}).get('filename')}")
                else:
                    print(f"Export failed: {result.get('message')}")
                continue
            elif user_input.lower() in ['debug_msg']:
                client.debug_msg = not client.debug_msg
                print(f"Debug message display: {'Enabled' if client.debug_msg else 'Disabled'}")
                continue

            elif user_input.lower() in ['train']:
                print("\nEntering training mode...")
                interactive_training_mode(client)
                print("Returned to main conversation mode")
                continue
                
            elif user_input.lower() in ['training options']:
                show_training_options(client)
                continue
                
            elif user_input.lower() in ['training status']:
                show_training_status(client)
                continue
                
            elif user_input.lower() in ['load model']:
                model_path = input("Please enter model file path (empty for default): ").strip()
                result = client.load_model(model_path if model_path else None)
                if result:
                    print("Model loaded successfully")
                else:
                    print(f"Error loading model: {result.get('message')}")
                continue

            elif user_input.lower() in ['create template']:
                file_name = input("Please enter template file name (default training_template.json): ").strip() or "training_template.json"
                result = client.create_training_template(file_name)
                if result:
                    print(f"Template created: {result.get('data', {}).get('file_path')}")
                else:
                    print(f"Error creating template: {result.get('message')}")
                continue

            elif user_input.lower() in ['help']:
                print_help()
                continue
                
            elif user_input.lower() in ['silent', 'quiet']:
                client.show_thinking = not client.show_thinking
                print(f"Thinking process display: {'Enabled' if client.show_thinking else 'Disabled'}")
                continue
            
            # Handle normal conversation
            response = client.chat(user_input)
            print(f"Serlin: {response}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupt signal detected, exiting...")
            summary_result = client.get_conversation_summary()
            if summary_result:
                print(f"\n{summary_result.get('data', {}).get('summary', 'No summary available')}")
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Please re-enter or type 'exit' to end conversation")

if __name__ == "__main__":
    main()