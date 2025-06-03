import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Send, Bot, User, Upload, X, FileText } from 'lucide-react';
import { Agent, ChatMessage } from '@/types/Agent';
import { ChatResponseCard } from './ChatResponseCard';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

interface ChatInterfaceProps {
  selectedAgent: Agent | null;
  conversationId: string | null;
  uploadedFile: File | null;
  onFileUpload: (file: File | null) => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  selectedAgent,
  conversationId,
  uploadedFile,
  onFileUpload,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationFile, setConversationFile] = useState<File | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (conversationId) {
      fetchMessages();
    } else {
      setMessages([]);
    }
  }, [conversationId]);

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const fetchMessages = async () => {
    if (!conversationId) return;
    
    try {
      const response = await fetch(`${API_URL}/api/conversations/${conversationId}/messages`);
      if (response.ok) {
        const data = await response.json();
        
        // Process the messages to handle different response formats
        const processedMessages = data.map((msg: any) => {
          // Handle custom agent responses that might contain execution_history
          if (msg.role === 'assistant' && typeof msg.content === 'string') {
            try {
              // Try to parse if content is JSON (from custom agents)
              const parsed = JSON.parse(msg.content);
              if (parsed.execution_history) {
                // Convert execution history to reasoning steps format
                const reasoning_steps = parsed.execution_history.map((exec: any, index: number) => ({
                  step_number: index + 1,
                  reasoning: exec.reasoning || '',
                  next_step: exec.next_step || '',
                  code: exec.code_to_execute || '',
                  output: exec.result?.output || '',
                  error: exec.result?.error || '',
                  plot_path: exec.result?.returned_objects?.plot_path || null
                }));
                
                return {
                  ...msg,
                  content: parsed.final_answer || 'Analysis completed.',
                  reasoning_steps,
                  plot_paths: parsed.execution_history
                    .map((exec: any) => exec.result?.returned_objects?.plot_path)
                    .filter(Boolean)
                };
              } else if (parsed.updated_answer || parsed.code) {
                // Handle custom agent direct responses
                return {
                  ...msg,
                  content: parsed.updated_answer || parsed.reasoning || msg.content,
                  code: parsed.code || undefined
                };
              }
            } catch (e) {
              // If not JSON, return as is
              return msg;
            }
          }
          return msg;
        });
        
        setMessages(processedMessages);
      }
    } catch (error) {
      console.error('Error fetching messages:', error);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setConversationFile(file);
    }
  };

  const removeConversationFile = () => {
    setConversationFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !conversationId || isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Upload file if present (prioritize conversation-specific file over global file)
      let file_path = null;
      const fileToUpload = conversationFile || uploadedFile;
      
      if (fileToUpload) {
        const formData = new FormData();
        formData.append('file', fileToUpload);
        
        const uploadResponse = await fetch(`${API_URL}/api/upload`, {
          method: 'POST',
          body: formData,
        });
        
        if (uploadResponse.ok) {
          const uploadData = await uploadResponse.json();
          file_path = uploadData.file_path;
          console.log('File uploaded successfully:', file_path);
        }
      }

      const response = await fetch(`${API_URL}/api/conversations/${conversationId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: inputValue,
          file_path: file_path,
        }),
      });

      if (response.ok) {
        // Refresh messages to get the AI response
        fetchMessages();
      }
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!selectedAgent) {
    return (
      <div className="flex-1 flex items-center justify-center bg-gray-900">
        <div className="text-center text-gray-400">
          <Bot className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <h2 className="text-xl font-medium mb-2">No Agent Selected</h2>
          <p>Select an agent from the sidebar to start chatting</p>
        </div>
      </div>
    );
  }

  if (!conversationId) {
    return (
      <div className="flex-1 flex items-center justify-center bg-gray-900">
        <div className="text-center text-gray-400">
          <Bot className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <h2 className="text-xl font-medium mb-2">No Conversation Selected</h2>
          <p>Start a new conversation to begin chatting with {selectedAgent.name}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-gray-900">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">{selectedAgent.name}</h2>
            <p className="text-sm text-gray-400">
              {selectedAgent.model_type} • Memory: {selectedAgent.memory_enabled ? 'On' : 'Off'}
            </p>
          </div>
          
          {/* Conversation-specific file upload */}
          <div className="flex items-center space-x-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.txt,.json"
              onChange={handleFileUpload}
              className="hidden"
            />
            <Button
              onClick={() => fileInputRef.current?.click()}
              variant="outline"
              size="sm"
              className="bg-gray-700 border-gray-600 hover:bg-gray-600"
            >
              <Upload className="w-4 h-4 mr-2" />
              Add File
            </Button>
            {(conversationFile || uploadedFile) && (
              <div className="text-sm text-gray-400 flex items-center">
                <FileText className="w-4 h-4 mr-1" />
                File: {(conversationFile || uploadedFile)?.name}
                {conversationFile && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={removeConversationFile}
                    className="h-4 w-4 p-0 ml-2 text-gray-400 hover:text-red-400"
                  >
                    <X className="w-3 h-3" />
                  </Button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
        <div className="space-y-4 max-w-4xl mx-auto">
          {messages.map((message, index) => (
            <div key={index}>
              {message.role === 'user' ? (
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                    <User className="w-4 h-4" />
                  </div>
                  <div className="flex-1 bg-gray-800 rounded-lg p-4">
                    <p className="text-white">{message.content}</p>
                  </div>
                </div>
              ) : message.role === 'assistant' ? (
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                    <Bot className="w-4 h-4" />
                  </div>
                  <div className="flex-1">
                    <ChatResponseCard message={message} />
                  </div>
                </div>
              ) : null}
            </div>
          ))}
          
          {isLoading && (
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                <Bot className="w-4 h-4" />
              </div>
              <div className="flex-1 bg-gray-800 rounded-lg p-4">
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-green-400"></div>
                  <p className="text-gray-400">Thinking...</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="border-t border-gray-700 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex space-x-2">
            <Textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={`Message ${selectedAgent.name}...`}
              className="flex-1 bg-gray-800 border-gray-600 text-white resize-none"
              rows={1}
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="bg-blue-600 hover:bg-blue-700"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};