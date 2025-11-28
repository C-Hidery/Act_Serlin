# Act_Server.py - Serlin Transformer Server
from time import sleep
import numpy as np
import random
import socket
import json
import threading
import datetime
import math


# Import Serlin modules
from Module.Serlin_Transformer import *


class SerlinServer:
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self.serlin = None
        self.server_socket = None
        self.running = False
        
    def initialize_serlin(self):
        """Initialize the Serlin model"""
        try:
            # Load configuration
            config_file = "serlin_config.json"
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    print(f"Loading parameters from configuration file {config_file}...")
                    
                    # Update SerlinConfig instance parameters
                    serlin_config.d_model = config_data.get('d_model', serlin_config.d_model)
                    serlin_config.nhead = config_data.get('nhead', serlin_config.nhead)
                    serlin_config.num_encoder_layers = config_data.get('num_encoder_layers', serlin_config.num_encoder_layers)
                    serlin_config.num_decoder_layers = config_data.get('num_decoder_layers', serlin_config.num_decoder_layers)
                    serlin_config.think_steps = config_data.get('think_steps', serlin_config.think_steps)
                    serlin_config.max_length = config_data.get('max_length', serlin_config.max_length)
                    
            except FileNotFoundError:
                print(f"Configuration file {config_file} not found, using default parameters")
            
            print("Initializing Transformer version Serlin system...")
            print("Current parameters:")
            print(f"  d_model: {serlin_config.d_model}")
            print(f"  nhead: {serlin_config.nhead}")
            print(f"  Encoder layers: {serlin_config.num_encoder_layers}")
            print(f"  Decoder layers: {serlin_config.num_decoder_layers}")
            print(f"  Thinking steps: {serlin_config.think_steps}")
            print(f"  Maximum sequence length: {serlin_config.max_length}")
            
            self.serlin = SerlinTransformer(
                d_model=serlin_config.d_model,
                nhead=serlin_config.nhead,
                num_encoder_layers=serlin_config.num_encoder_layers,
                num_decoder_layers=serlin_config.num_decoder_layers,
                think_steps=serlin_config.think_steps,
                max_length=serlin_config.max_length
            )
            
            print("Serlin system initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Error initializing Serlin: {e}")
            return False
    
    def handle_request(self, request_data):
        """Handle client requests"""
        try:
            user_id = request_data.get('user_id', 'default_user')
            command = request_data.get('command', '')
            data = request_data.get('data', {})
            
            if command == 'chat':
                user_input = data.get('user_input', '')
                show_thinking = data.get('show_thinking', False)
                
                if user_input:
                    response = self.serlin.chat(user_id, user_input, show_thinking=show_thinking)
                    return {
                        "status": "success",
                        "message": "Chat response generated",
                        "data": {
                            "response": response,
                            "user_id": user_id
                        }
                    }
                else:
                    return {"status": "error", "message": "No user input provided"}
                    
            elif command == 'get_system_status':
                #print("getting")
                status = self.serlin.get_system_status(user_id)
                #print("Get, returning")
                return {
                    "status": "success",
                    "message": "System status retrieved",
                    "data": status
                }
                
            elif command == 'get_conversation_summary':
                summary = self.serlin.get_conversation_summary()
                return {
                    "status": "success",
                    "message": "Conversation summary retrieved",
                    "data": {"summary": summary}
                }
                
            elif command == 'add_training_data':
                questions = data.get('questions', [])
                answers = data.get('answers', [])
                self.serlin.add_training_data(questions, answers)
                return {
                    "status": "success",
                    "message": f"Added {len(questions)} training pairs",
                    "data": {"training_data_count": len(self.serlin.training_data)}
                }
                
            elif command == 'train':
                epochs = data.get('epochs', 100)
                batch_size = data.get('batch_size', 4)
                learning_rate = data.get('learning_rate', 0.0001)
                early_stopping = data.get('early_stopping', True)
                
                # Update learning rate
                if hasattr(self.serlin.trainer.optimizer, 'param_groups'):
                    for param_group in self.serlin.trainer.optimizer.param_groups:
                        param_group['lr'] = learning_rate
                
                result = self.serlin.improved_train(
                    epochs=epochs,
                    batch_size=batch_size,
                    early_stopping=early_stopping
                )
                return result
                
            elif command == 'validate_training_data':
                result = self.serlin.validate_training_data()
                return result
                
            elif command == 'get_vocabulary_info':
                result = self.serlin.get_vocabulary_info()
                return result
                
            elif command == 'expand_vocabulary':
                words_text = data.get('words_text', '')
                result = self.serlin.expand_vocabulary(words_text)
                return result
                
            elif command == 'reload_vocabulary':
                result = self.serlin.reload_vocabulary()
                return result
                
            elif command == 'fix_model_vocab':
                result = self.serlin.fix_model_vocab()
                return result
                
            elif command == 'reset_model':
                result = self.serlin.reset_model()
                return result
                
            elif command == 'export_conversation':
                filename = data.get('filename')
                result = self.serlin.export_conversation(filename)
                return {
                    "status": "success" if result else "error",
                    "message": "Conversation exported" if result else "Export failed",
                    "data": {"filename": result}
                }
                
            elif command == 'load_model':
                path = data.get('path')
                result = self.serlin.load_model(path)
                return result
                
            elif command == 'save_model':
                path = data.get('path')
                result = self.serlin.save_model(path)
                return {
                    "status": "success",
                    "message": "Model saved",
                    "data": {"path": result}
                }
                
            elif command == 'create_training_template':
                #file_path = data.get('file_path', 'training_template.json')
                result = self.serlin.create_training_template()
                return {
                    "status": "success",
                    "message": "Training template send.",
                    "data": result
                }
                
            elif command == 'batch_train_from_json':
                file_path = data.get('training_data')
                if file_path:
                    result = self.serlin.batch_train_from_json(file_path)
                    return result
                else:
                    return {"status": "error", "message": "No file path provided"}
                    
            else:
                return {"status": "error", "message": f"Unknown command: {command}"}
                
        except Exception as e:
            return {"status": "error", "message": f"Error handling request: {str(e)}"}
    def send_rep(self,client_socket,data=None):
        try:
            # Send response
            response_json = json.dumps(data).encode('utf-8')
            response_length = len(response_json)
                
                    #  ȷ     Ӧ    
            client_socket.send(response_length.to_bytes(4, byteorder='big'))
                    #  ٷ     Ӧ    
            bytes_sent = 0
            while bytes_sent < response_length:
                sent = client_socket.send(response_json[bytes_sent:bytes_sent+4096])
                if sent == 0:
                    break
                bytes_sent += sent
                
            print(f"Response sent ({bytes_sent}/{response_length} bytes)")
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            error_response = {
                "status": "error",
                "message": "Invalid JSON format"
            }
            error_json = json.dumps(error_response).encode('utf-8')
            error_length = len(error_json)
            client_socket.send(error_length.to_bytes(4, byteorder='big'))
            client_socket.send(error_json)
        except Exception as e:
            print(f"Error Sending response: {e}")
    def recv_req(self,client_socket):
        try:
            #  Ƚ      ݳ   
            length_data = client_socket.recv(4)
            if not length_data:
                return None
        
            request_length = int.from_bytes(length_data, byteorder='big')
            print(f"Expecting request of {request_length} bytes")
        
            #             
            request_data = b""
            bytes_received = 0
        
            while bytes_received < request_length:
                chunk = client_socket.recv(min(4096, request_length - bytes_received))
                if not chunk:
                    break
                request_data += chunk
                bytes_received += len(chunk)
                print(f"Received {bytes_received}/{request_length} bytes")
        
            if bytes_received < request_length:
                print(f"Incomplete request: {bytes_received}/{request_length} bytes")
                return None
        
            try:
                # Parse JSON request
                request_data_json = request_data.decode('utf-8')
                request_data_obj = json.loads(request_data_json)
                print(f"Received request: {request_data_obj.get('command', 'unknown')}")
                return request_data_obj
            
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return None
        except Exception as e:
            print(f"Error receiving request: {e}")
            return None
    def handle_client(self, client_socket, address):
        """Handle individual client connection"""
        print(f"Connection from {address}")
    
        try:
            while True:
                request_data_obj = self.recv_req(client_socket)
                if not request_data_obj:
                    #print(f"No valid request data from {address}, closing connection.")
                    break
                print(f"Received request: {request_data_obj.get('command', 'unknown')}")
                # Process request
                response = self.handle_request(request_data_obj)
                
                self.send_rep(client_socket,response)
                
               
                
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
            print(f"Connection with {address} closed")
    
    def start_server(self):
        """Start the server"""
        if not self.initialize_serlin():
            print("Failed to initialize Serlin system")
            return
    
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
        try:
            print(f"Attempting to bind to {self.host}:{self.port}...")
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
        
            print(f"Serlin Server successfully started on {self.host}:{self.port}")
            print("Waiting for connections...")
        
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"New connection from {address}")
                
                    # Handle client in a new thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                
                except KeyboardInterrupt:
                    print("\nServer shutdown requested...")
                    break
                except Exception as e:
                    print(f"Error accepting connection: {e}")
                
        except OSError as e:
            print(f"Failed to start server: {e}")
            if "Address already in use" in str(e):
                print(f"Port {self.port} is already in use. Please use a different port.")
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
            print("Server stopped")

def main():
    # Get server configuration
    host = input("Enter server host (default: localhost): ").strip() or 'localhost'
    try:
        port = int(input("Enter server port (default: 5555): ").strip() or '5555')
    except ValueError:
        port = 5555
        print("Invalid port, using default: 5555")
    
    # Start server
    server = SerlinServer(host, port)
    server.start_server()

if __name__ == "__main__":
    main()